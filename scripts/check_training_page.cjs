const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

module.exports = async function checkTrainingPage(page, address, root, output) {
  const list = await (await page.request.get(address + 'api/training/runs')).json();
  assert.ok(list.length, 'Run training first');
  const name = list[0].name;
  const directory = path.join(root, 'runs/training', name);
  const best = JSON.parse(fs.readFileSync(path.join(directory, 'best.json')));
  const decoder = JSON.parse(fs.readFileSync(path.join(directory, 'decoder.json')));
  const binding = JSON.parse(fs.readFileSync(path.join(root, 'data/brain/compiled/output-binding.json')));
  const body = JSON.parse(fs.readFileSync(path.join(root, 'config/body-map.json')));
  const { BrainBodyBridge } = await import(pathToFileURL(path.join(root, 'src/bridge/controller.mjs')));
  const bridge = new BrainBodyBridge(binding, body, decoder);
  bridge.reset(best.frames[0].jointAngles);
  for (const frame of best.frames.slice(1)) {
    const ratesHz = Object.fromEntries(binding.neurons.map(neuron => [neuron.id, 0]));
    for (const [index, count] of frame.brain.outputCounts) ratesHz[binding.neurons[index].id] = count / 0.02;
    const result = bridge.step({
      bindingId: binding.id, dataset: 'flywire-783', encoding: 'dense', ratesHz,
    }, 0.02);
    result.actions.forEach((value, i) => assert.ok(Math.abs(value - frame.decoderActions[i]) < 1e-10));
    result.angles.forEach((value, i) => assert.ok(Math.abs(value - frame.nextCommandAngles[i]) < 1e-10));
  }
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto(`${address}demo/training.html?capture=1&run=${encodeURIComponent(name)}`, {
    waitUntil: 'domcontentloaded', timeout: 120000,
  });
  await page.waitForFunction(() => window.trainingReady || window.trainingError, null, { timeout: 120000 });
  assert.equal(await page.evaluate(() => window.trainingError), undefined);
  await page.waitForFunction(() => document.querySelector('#replay').contentWindow?.sensoryReady, null, { timeout: 120000 });
  const status = await page.evaluate(() => {
    const child = document.querySelector('#replay').contentWindow;
    child.setSensoryTime(0.8);
    return child.getSensoryStatus();
  });
  assert.equal(status.brainConnected, true);
  assert.equal(status.supported, false);
  assert.deepEqual(status.sample, best.frames[40]);
  assert.equal(status.visualModel, 'original-physics-primitives');
  assert.deepEqual(status.geometryPoses, status.sample.geometryPoses);
  assert.equal((await page.request.get(address + 'assets/models/sakura-miko-1/model.pmx')).status(), 404);
  assert.ok(best.metrics.outputSpikeCount > 0);
  assert.equal(best.brainModel.neuronCount, 138639);
  const activityFile = path.join(directory, 'best-activity.json');
  if (fs.existsSync(activityFile)) {
    const activity = JSON.parse(fs.readFileSync(activityFile));
    await page.waitForFunction(() => Math.abs(window.neuralActivityStatus?.timeSeconds - .8) < 1e-7, null, { timeout: 120000 });
    const actual = await page.evaluate(() => window.neuralActivityStatus);
    assert.equal(actual.neuronCount, best.brainModel.neuronCount);
    assert.equal(actual.spikeCount, best.frames[40].brain.spikeCount);
    assert.equal(actual.activeCount, activity.frames[40].counts.length);
    await page.locator('#activity-role').selectOption('output');
    const outputActivity = await page.evaluate(() => window.neuralActivityStatus);
    assert.equal(outputActivity.spikeCount, best.frames[40].brain.outputSpikeCount);
    const selectedIndex = activity.frames[40].counts[0][0];
    await page.locator('#neuron-search').fill(activity.neuronIds[selectedIndex]);
    await page.locator('#neuron-find').click();
    assert.equal(await page.evaluate(() => window.neuralActivityStatus.selected), selectedIndex);
    await page.locator('#activity-role').selectOption('all');

    await page.locator('#activity-card').screenshot({ path: path.join(output, 'brain-activity.png') });
  }
  assert.equal(name, 'random-head-bias-02');
  assert.equal(best.task.id, 'neutral-head-height-v4-bias');
  assert.equal(best.task.initialPose, 'random');
  assert.equal(best.readout.trainBias, true);
  assert.deepEqual(best.rigPoseOffsets, {});
  assert.equal(best.channels.length, 126);
  assert.match(await page.locator('#task-note').textContent(), /ランダムな初期姿勢2通り/);
  const second = JSON.parse(fs.readFileSync(path.join(directory, 'best-second.json')));
  assert.notDeepEqual(second.initialState.qpos, best.initialState.qpos);
  assert.equal(best.objectiveSummary.return, (best.metrics.return + second.metrics.return) / 2);
  await page.evaluate(async () => {
    await document.fonts.ready;
    await document.querySelector('#replay').contentWindow.document.fonts.ready;
    await new Promise(requestAnimationFrame);
  });
  await page.screenshot({ path: path.join(output, 'physics-training-dashboard.png'), timeout: 120000 });
  const child = page.frames().find(frame => frame.url().includes('/demo/sensory.html?'));
  await child.locator('#scenario').selectOption('baseline');
  const baseline = JSON.parse(fs.readFileSync(path.join(directory, 'baseline.json')));
  await page.waitForFunction(fingerprint => {
    const child = document.querySelector('#replay').contentWindow;
    return child.getSensoryRecording()?.policyFingerprint === fingerprint;
  }, baseline.policyFingerprint);
  const baselineStatus = await page.evaluate(() => {
    const child = document.querySelector('#replay').contentWindow;
    child.setSensoryTime(.8); return child.getSensoryStatus();
  });
  assert.deepEqual(baselineStatus.sample, baseline.frames[40]);
  await child.locator('#scenario').selectOption('best-second');
  await page.waitForFunction(seed => document.querySelector('#replay').contentWindow?.getSensoryRecording()?.initialState?.seed === seed,
    second.initialState.seed);
  const state2 = await page.evaluate(() => {
    const child = document.querySelector('#replay').contentWindow;
    child.setSensoryTime(.8); return child.getSensoryStatus();
  });
  assert.deepEqual(state2.geometryPoses, second.frames[40].geometryPoses);
  assert.equal((await page.request.get(address + 'api/training/crawl-brain-drive-01/status')).status(), 404);
  assert.equal((await page.request.get(address + 'assets/local-preview/avatar.js')).status(), 404);
  assert.equal((await page.request.get(address + '.env')).status(), 404);
  assert.equal((await page.request.get(address + `api/training/${name}/latest.npz`)).status(), 404);
  assert.equal((await page.request.post(address + 'api/training/start')).status(), 405);
  assert.deepEqual(errors, []);
  console.log('Verified real-brain replay, baseline switch, live status, API boundary, and Python/JS decoder parity');
};
