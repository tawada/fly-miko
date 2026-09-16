import { ActivityView } from './neural-activity.js';
const $ = id => document.getElementById(id);
const activityView = new ActivityView();
let currentRun = new URLSearchParams(location.search).get('run');
let status, activeRecording, spikeRows = [], replayLoadedFor = null, replayBestReturn, drawnRecording, drawnTime;
const statuses = { initializing: '初期化中', training: '学習中', evaluating: '検証中', complete: '完了', interrupted: '中断', failed: '失敗' };
async function json(url) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw Error(`${url}: ${response.status}`);
  return response.json();
}
function context(id) {
  const canvas = $(id), w = Math.max(1, canvas.clientWidth), h = Math.max(1, canvas.clientHeight);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}
function drawLearning() {
  const { ctx, w, h } = context('learning');
  const history = status?.history || [];
  if (!history.length) return;
  const low = Math.min(0, ...history.map(row => row.meanReturn));
  const high = Math.max(low + 0.1, ...history.map(row => row.bestReturn)) + 0.1;
  ctx.font = '10px sans-serif'; ctx.fillStyle = '#9d8ba4';
  ctx.fillText(high.toFixed(1), 0, 12); ctx.fillText(low.toFixed(1), 0, h - 3);
  for (const [field, color] of [['meanReturn', '#bfb3c7'], ['bestReturn', '#ad72b0']]) {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    history.forEach((row, i) => {
      const x = 28 + i / Math.max(1, history.length - 1) * (w - 32);
      const y = h - 10 - (row[field] - low) / (high - low) * (h - 22);
      if (!i) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }); ctx.stroke();
    if (history.length === 1) {
      ctx.fillStyle = color; ctx.beginPath();
      ctx.arc(28, h - 10 - (history[0][field] - low) / (high - low) * (h - 22), 3, 0, 2 * Math.PI); ctx.fill();
    }
  }
}
function updateEvaluation(rows) {
  const heightTask = status?.config.task?.objective === 'integral-head-centre-height';
  $('metric-heading').textContent = heightTask ? '平均頭高 m' : '進行 m';
  const labels = { untrained: '初期パラメータ', trained: '学習後', 'brain-output-ablated': '脳出力遮断' };
  $('evaluation').replaceChildren();
  for (const [key, label] of Object.entries(labels)) {
    const group = rows.filter(row => row.policy === key);
    const row = document.createElement('tr');
    const average = field => group.length && group.every(row => Number.isFinite(row[field]))
      ? (group.reduce((sum, r) => sum + r[field], 0) / group.length).toFixed(field === 'meanAbsNeuralAction' ? 3 : 2) : '—';
    for (const value of [label + (group.length ? ` (${group.length})` : ''), average('return'), average('survivalSeconds'), average(heightTask ? 'meanHeadHeightM' : 'forwardDistanceM'), average('meanAbsNeuralAction')]) {
      const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
    }
    $('evaluation').append(row);
  }
}
async function poll() {
  if (!currentRun) return;
  status = await json(`/api/training/${encodeURIComponent(currentRun)}/status`);
  const { config, progress } = status;
  $('network').textContent = `${(config.brainNeuronCount / 1000).toFixed(1)}k / ${(config.brainConnectionCount / 1e6).toFixed(1)}M`;
  $('generation').textContent = `${progress?.generation ?? 0} / ${progress?.targetGenerations ?? '—'}`;
  $('best-return').textContent = progress?.bestReturn?.toFixed(3) ?? '—';
  $('status').textContent = statuses[progress?.status] || '準備中';
  $('evaluation-title').textContent = config.objectiveEpisodeCount === 2 ? '目的関数と同じ初期状態2通りで比較' : `別の${config.evaluationSeedCount ?? 3} seedで比較`;
  $('details').textContent = `${config.outputNeuronCount}出力 → ${config.featureCount}特徴 → 18関節。${config.parameterCount}パラメータ。評価 ${progress?.completedEvaluations ?? 0}回。${config.neuralDtMs}ms刻みの脳計算。`;
  $('task-note').textContent = config.objectiveEpisodeCount === 2
    ? `ランダムな初期姿勢2通りの平均で評価。各${config.episodeSeconds}秒、頭の高さを加点（m・秒）。1世代10候補、計算済み評価は再利用。再生は初期状態1（切替で2）。`
    : config.task?.objective === 'integral-head-centre-height'
    ? `脚を開いた四つん這いから開始。${config.episodeSeconds}秒間、頭の中心が高いほど加点（m・秒）。移動の加点なし。低い姿勢でも試行を継続。`
    : `立位から開始。${config.episodeSeconds}秒間、姿勢維持と前進速度を評価。`;
  updateEvaluation(status.evaluation);
  drawLearning();
  window.trainingStatus = status;
  window.trainingReady = true;
  for (const option of $('replay').contentWindow?.document.querySelectorAll('#scenario option') ?? []) {
    option.disabled = !status.recordingsAvailable?.[option.value];
  }
  const showingBest = $('replay').contentWindow?.document.querySelector('#scenario')?.value === 'best';
  if (status.recordingsAvailable?.best && (replayLoadedFor !== currentRun
      || (showingBest && replayBestReturn !== progress?.bestReturn))) loadReplay();
}
function loadReplay() {
  if (!currentRun || !status.recordingsAvailable?.best) return;
  const params = new URLSearchParams({ run: currentRun, embed: '1' });
  if (new URLSearchParams(location.search).has('capture')) params.set('capture', '1');
  $('replay').src = `/demo/sensory.html?${params}`;
  replayLoadedFor = currentRun;
  replayBestReturn = status.progress?.bestReturn;
  activeRecording = null;
}
function drawSpikes(recording, time) {
  if (activeRecording !== recording) {
    activeRecording = recording;
    const totals = new Map();
    for (const frame of recording.frames) for (const [index, count] of frame.brain?.outputCounts || []) {
      totals.set(index, (totals.get(index) || 0) + count);
    }
    spikeRows = [...totals].sort((a, b) => b[1] - a[1]).slice(0, 16).map(([index]) => index);
  }
  const { ctx, w, h } = context('spikes');
  const left = 80, rowHeight = h / Math.max(1, spikeRows.length);
  const rowIndex = new Map(spikeRows.map((index, i) => [index, i]));
  ctx.font = '9px sans-serif'; ctx.fillStyle = '#9d8ba4';
  spikeRows.forEach((index, row) => {
    const neuron = recording.outputNeurons[index];
    ctx.fillText((neuron.cellType || neuron.id.slice(-8)).slice(0, 12), 0, (row + 0.75) * rowHeight);
  });
  ctx.fillStyle = '#7f99ca';
  for (const frame of recording.frames) for (const [index, count] of frame.brain?.outputCounts || []) {
    if (rowIndex.has(index)) {
      const x = left + frame.timeSeconds / recording.durationSeconds * (w - left);
      ctx.fillRect(x, rowIndex.get(index) * rowHeight + 1, Math.min(5, 1 + count), Math.max(2, rowHeight - 2));
    }
  }
  const x = left + time / recording.durationSeconds * (w - left);
  ctx.strokeStyle = '#b66292'; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
}
function drawHeight(recording, time) {
  const available = Number.isFinite(recording.frames[0].headHeightM);
  $('height-card').hidden = !available;
  if (!available) return;
  const { ctx, w, h } = context('height-chart');
  const top = Math.max(1.5, ...recording.frames.map(frame => frame.headHeightM));
  ctx.fillStyle = '#9d8ba4'; ctx.font = '10px sans-serif';
  ctx.fillText(`${top.toFixed(1)} m`, 0, 12); ctx.fillText('0', 0, h - 3);
  ctx.strokeStyle = '#ad72b0'; ctx.lineWidth = 2; ctx.beginPath();
  recording.frames.forEach((frame, i) => {
    const x = 35 + frame.timeSeconds / recording.durationSeconds * (w - 40);
    const y = h - 10 - frame.headHeightM / top * (h - 24);
    if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
  });
  ctx.stroke();
  const x = 35 + time / recording.durationSeconds * (w - 40);
  ctx.strokeStyle = '#7f99ca'; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
}
function animate() {
  const child = $('replay').contentWindow;
  if (child?.sensoryReady) {
    const state = child.getSensoryStatus();
    const recording = child.getSensoryRecording();
    activityView.update(currentRun, child.document.querySelector('#scenario').value, recording, state.sample.timeSeconds);
    if (drawnRecording === recording && drawnTime === state.sample.timeSeconds) {
      requestAnimationFrame(animate);
      return;
    }
    drawnRecording = recording; drawnTime = state.sample.timeSeconds;
    $('policy-note').textContent = recording.ablateBrain
      ? '脳出力を遮断した比較です。脳の計算は続けますが、関節へは固定の基準姿勢だけを指令します。'
      : recording.neuralDriveEnabled
      ? `脳駆動あり。${recording.readout?.trainBias === false ? '学習するバイアスなし。四つん這いを基準に脳信号で関節を動かします。' : '32特徴量の重みと18関節のバイアスを学習します。'}`
      : recording.learned
      ? '学習済み方策の再生です。脳の寄与は「脳出力遮断」との比較で確認してください。'
      : '初期方策の再生です。脳は計算していますが、脳から関節への重みはゼロです。動きは固定した姿勢指令と重力によるものです。';
    const brain = state.sample.brain;
    $('brain-spikes').textContent = brain?.spikeCount ?? 0;
    $('output-spikes').textContent = brain?.outputSpikeCount ?? 0;
    drawSpikes(recording, state.sample.timeSeconds);
    drawHeight(recording, state.sample.timeSeconds);
    $('head-height').textContent = Number.isFinite(state.sample.headHeightM) ? `${state.sample.headHeightM.toFixed(3)} m` : '';
    const ids = state.sample.bodyContactIds;
    $('body-input-card').hidden = !ids || recording.channels.length !== 126;
    if (ids) {
      $('body-inputs').replaceChildren();
      for (const [part, label] of [['hand', '手'], ['upper_arm', '腕1（上腕）'], ['forearm', '腕2（前腕）'],
        ['foot', '足'], ['thigh', '脚1（大腿）'], ['shin', '脚2（下腿）']]) {
        const row = document.createElement('tr');
        const values = [label, ...['left', 'right'].map(side => {
          const i = ids.indexOf(`${side}_${part}`);
          return `${state.sample.bodyContacts[i] ? '● ' : ''}${state.sample.bodyContactForcesN[i].toFixed(1)}`;
        })];
        for (const value of values) { const cell = document.createElement('td'); cell.textContent = value; row.append(cell); }
        $('body-inputs').append(row);
      }
      $('gravity-input').textContent = `重力（身体座標 x/y/z）：${state.sample.projectedGravity.map(v => v.toFixed(2)).join(' / ')}`;
    }
  }
  requestAnimationFrame(animate);
}
addEventListener('resize', () => { drawnTime = null; drawLearning(); });
try {
  const runs = await json('/api/training/runs');
  if (!runs.length) {
    $('empty').style.display = 'block';
    $('refresh').onclick = () => location.reload();
  } else {
    for (const run of runs) {
      const option = document.createElement('option'); option.value = run.name; option.textContent = run.name;
      $('run').append(option);
    }
    if (!runs.some(run => run.name === currentRun)) currentRun = runs[0].name;
    $('run').value = currentRun;
    await poll();
    $('run').onchange = async event => { currentRun = event.target.value; replayLoadedFor = null; await poll(); };
    $('refresh').onclick = async () => { await poll(); loadReplay(); };
    animate();
  }
  setInterval(async () => {
    try {
      if (currentRun) await poll();
      else if ((await json('/api/training/runs')).length) location.reload();
    } catch (error) { $('error').textContent = String(error); }
  }, 1500);
} catch (error) {
  window.trainingError = String(error); $('error').textContent = String(error);
}
