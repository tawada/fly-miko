const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { spawn } = require('node:child_process');
const { once } = require('node:events');
const ffmpeg = require('ffmpeg-static');

module.exports = async function renderTraining(page, address, root, output) {
  const runs = await (await page.request.get(address + 'api/training/runs')).json();
  const pointer = path.join(root, 'runs/training/latest.json');
  const latestName = fs.existsSync(pointer) ? JSON.parse(fs.readFileSync(pointer)).name : null;
  const run = runs.find(run => run.name === latestName) || runs[0];
  if (!run) throw Error('No saved training run');
  const name = run.name;
  const directory = path.join(root, 'runs/training', name);
  const source = fs.readFileSync(path.join(directory, 'best.json'));
  const recording = JSON.parse(source);
  if (!recording.brainConnected || !recording.frames.length) throw Error('No neural replay available');
  const snapshot = await (await page.request.get(address + `api/training/${name}/status`)).json();
  const generation = snapshot.progress.generation;
  const fps = 24, seconds = recording.durationSeconds;
  const frameCount = Math.round(seconds * fps);
  const stem = `training-${name}-generation-${generation}-physics`;
  const movie = path.join(output, `${stem}.mp4`);
  const poster = path.join(output, `${stem}.png`);
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));

  // Freeze the saved result and status while producing frames; live training
  // must not switch policies or reset playback halfway through this movie.
  await page.route('**/api/training/**', async route => {
    const url = new URL(route.request().url());
    const value = url.pathname === '/api/training/runs' ? runs
      : url.pathname === `/api/training/${name}/status` ? snapshot
      : url.pathname === `/api/training/${name}/best` ? recording : undefined;
    if (value === undefined) return route.continue();
    await route.fulfill({ json: value });
  });
  await page.goto(`${address}demo/training.html?capture=1&run=${encodeURIComponent(name)}`, {
    waitUntil: 'domcontentloaded', timeout: 120000,
  });
  await page.waitForFunction(() => window.trainingReady || window.trainingError, null, { timeout: 120000 });
  const problem = await page.evaluate(() => window.trainingError);
  if (problem) throw Error(problem);
  await page.waitForFunction(() => document.querySelector('#replay').contentWindow?.sensoryReady,
    null, { timeout: 120000 });

  // A capture-only layout keeps all of the recorded evidence in a 1080p frame.
  await page.addStyleTag({ content: `
    header{padding:14px 24px} h1{font-size:23px}
    .stats{padding:10px 24px;gap:12px}.stat{padding:9px 12px}.stat strong{font-size:23px}
    main{grid-template-columns:minmax(0,1fr) 680px;gap:16px;padding-bottom:0}
    iframe{height:770px}main>section{min-width:0}
    aside{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:160px 330px 270px;gap:10px;align-content:start}
    aside>.card{margin:0;padding:12px;overflow:hidden}
    aside>.card:nth-child(5){grid-column:1 / -1}
    aside>.card:nth-child(6){display:none}
    .chart{height:75px}#spikes{height:240px}
    .small{font-size:11px}.note{font-size:12px;margin:8px 4px}
    #body-input-card table{font-size:12px}#body-input-card td,#body-input-card th{padding:6px 3px}
    #evaluation{font-size:13px}#evaluation td{padding:12px 3px}
    #snapshot-label{font-size:12px;color:#87798a;text-align:right;line-height:1.7}
  ` });
  await page.evaluate(({ name, generation }) => {
    const controls = document.querySelector('header .controls');
    controls.replaceChildren();
    const label = document.createElement('div');
    label.id = 'snapshot-label';
    label.textContent = `${name} ／ 保存済み ${generation}世代目の再生`;
    controls.append(label);
  }, { name, generation });
  await page.evaluate(async () => {
    await document.fonts.ready;
    await document.querySelector('#replay').contentWindow.document.fonts.ready;
    await new Promise(requestAnimationFrame);
  });
  const setTime = async time => page.evaluate(async time => {
    const child = document.querySelector('#replay').contentWindow;
    child.setSensoryTime(time);
    // Let the parent update neural plots and measurements for the same frame.
    await new Promise(requestAnimationFrame);
    await new Promise(requestAnimationFrame);
    return child.getSensoryStatus().sample.timeSeconds;
  }, time);
  await setTime(0);
  await page.screenshot({ path: poster, timeout: 120000 });

  const encoder = spawn(ffmpeg, [
    '-y', '-loglevel', 'error', '-f', 'image2pipe', '-vcodec', 'png',
    '-framerate', String(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
    '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', movie,
  ], { stdio: ['pipe', 'ignore', 'pipe'] });
  let encoderError = '';
  encoder.stderr.on('data', chunk => { encoderError += chunk; });
  encoder.stdin.on('error', () => {});
  const finished = once(encoder, 'close');
  try {
    for (let frame = 0; frame < frameCount; frame++) {
      const time = frame === frameCount - 1 ? seconds : frame / fps;
      const measuredTime = await setTime(time);
      if (Math.abs(measuredTime - time) > recording.controlDtSeconds + 1e-6) throw Error('Replay clock mismatch');
      const image = await page.screenshot({ timeout: 120000 });
      if (encoder.exitCode !== null || encoder.stdin.destroyed) throw Error(encoderError || 'Encoder stopped');
      if (!encoder.stdin.write(image)) await once(encoder.stdin, 'drain');
      if (frame % fps === 0) console.log(`Training video: ${frame / fps}/${seconds.toFixed(0)} seconds`);
    }
    encoder.stdin.end();
    const [code] = await finished;
    if (code !== 0) throw Error(encoderError);
    if (errors.length) throw Error(errors.join('\n'));
    fs.writeFileSync(path.join(output, `${stem}.json`), JSON.stringify({
      run: name, generation, capturedAt: new Date().toISOString(),
      sourceRecording: 'best.json', sourceSha256: crypto.createHash('sha256').update(source).digest('hex'),
      fps, frameCount, durationSeconds: frameCount / fps, width: 1920, height: 1080,
      metrics: recording.metrics, evaluation: snapshot.evaluation,
      note: 'Replay of a saved physical trajectory, not live training or generated choreography.',
    }, null, 2));
    console.log(`Saved ${movie}`);
  } finally {
    if (encoder.exitCode === null) encoder.kill();
  }
};
