import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = id => document.getElementById(id);
const query = new URLSearchParams(location.search);
const trainingRun = query.get('run') || 'random-head-bias-02';
if (trainingRun && !/^[A-Za-z0-9_-]{1,80}$/.test(trainingRun)) throw Error('Invalid training run');
if (query.has('embed')) document.body.classList.add('embedded');
const stage = $('stage');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
stage.prepend(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color('#f9eeee');
scene.fog = new THREE.Fog('#f9eeee', 6, 16);
const camera = new THREE.PerspectiveCamera(33, 1, 0.01, 50);
camera.position.set(1.35, 1.1, 3.25);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.78, 0);
controls.enableDamping = false;
controls.update();
scene.add(new THREE.HemisphereLight(0xfff9fb, 0xb4a0b4, 1.1));
const light = new THREE.DirectionalLight(0xfff5eb, 0.9);
light.position.set(-2, 3, 4);
scene.add(light);
const floor = new THREE.Mesh(new THREE.PlaneGeometry(30, 30), new THREE.MeshBasicMaterial({ color: '#f2e7ed' }));
floor.rotation.x = -Math.PI / 2;
floor.position.y = -0.003;
scene.add(floor);
const grid = new THREE.GridHelper(8, 40, '#ddc6d0', '#ead9e1');
grid.position.y = -0.001;
scene.add(grid);
const collisionGroup = new THREE.Group();
collisionGroup.visible = true;
scene.add(collisionGroup);
const support = new THREE.Mesh(new THREE.TorusGeometry(0.14, 0.006, 8, 48),
  new THREE.MeshBasicMaterial({ color: '#b79ba9' }));
support.rotation.x = Math.PI / 2;
scene.add(support);
function resize() {
  renderer.setSize(stage.clientWidth, stage.clientHeight);
  camera.aspect = stage.clientWidth / stage.clientHeight;
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);
resize();

async function json(url) {
  const response = await fetch(url);
  if (!response.ok) throw Error(`${url} を読めません。学習の初期評価が完了しているか確認してください。`);
  return response.json();
}
let recording, body, sample, index = 0, time = 0, playing = true;
let pending = false;

function canvasContext(id) {
  const canvas = $(id);
  const width = Math.max(1, canvas.clientWidth);
  const height = Math.max(1, canvas.clientHeight);
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}
function marker(ctx, width, height) {
  const x = time / recording.durationSeconds * width;
  ctx.strokeStyle = '#705c74';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
}
function drawCharts() {
  let { ctx, width, height } = canvasContext('contact-chart');
  for (let side = 0; side < 2; side++) {
    ctx.fillStyle = '#f3edef';
    ctx.fillRect(0, 8 + side * 26, width, 17);
    ctx.fillStyle = side === 0 ? '#34a79d' : '#80c2b8';
    for (let i = 0; i < recording.frames.length - 1; i++) {
      if (recording.frames[i].footContacts[side]) {
        const start = recording.frames[i].timeSeconds / recording.durationSeconds * width;
        const end = recording.frames[i + 1].timeSeconds / recording.durationSeconds * width;
        ctx.fillRect(start, 8 + side * 26, Math.max(1, end - start), 17);
      }
    }
  }
  marker(ctx, width, height);
  ({ ctx, width, height } = canvasContext('joint-chart'));
  const j = Number($('joint').value);
  const values = recording.frames.flatMap(f => [f.jointAngles[j], f.commandAngles[j]]);
  const low = Math.min(-0.05, ...values) - 0.05;
  const high = Math.max(0.3, ...values) + 0.05;
  const y = value => height - 12 - (value - low) / (high - low) * (height - 24);
  ctx.strokeStyle = '#f0e9ed'; ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    ctx.beginPath(); ctx.moveTo(0, height * i / 4); ctx.lineTo(width, height * i / 4); ctx.stroke();
  }
  for (const [field, color] of [['commandAngles', '#be86aa'], ['jointAngles', '#1b958e']]) {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    recording.frames.forEach((frame, i) => {
      const x = frame.timeSeconds / recording.durationSeconds * width;
      if (i === 0) ctx.moveTo(x, y(frame[field][j])); else ctx.lineTo(x, y(frame[field][j]));
    });
    ctx.stroke();
  }
  marker(ctx, width, height);
  $('joint-value').textContent = `${sample.jointAngles[j].toFixed(3)} rad`;
  ({ ctx, width, height } = canvasContext('receptor-chart'));
  sample.sensory.values.forEach((value, i) => {
    const x = i / recording.channels.length * width;
    const barWidth = Math.max(1, width / recording.channels.length - 1);
    ctx.fillStyle = '#f2edf2'; ctx.fillRect(x, 8, barWidth, height - 14);
    ctx.fillStyle = i < 72 ? '#b88cc7' : i < 120 ? '#34a79d' : '#7f99ca';
    ctx.fillRect(x, height - 6 - value * (height - 14), barWidth, value * (height - 14));
  });
}

function showFrame(t) {
  if (!recording || pending) return;
  time = Math.max(0, Math.min(t, recording.durationSeconds));
  index = Math.min(recording.frames.length - 1, Math.floor(time / recording.controlDtSeconds + 1e-7));
  sample = recording.frames[index];
  collisionGroup.children.forEach((shape, i) => {
    shape.position.fromArray(sample.geometryPoses[i].position);
    shape.quaternion.fromArray(sample.geometryPoses[i].quaternion);
  });
  support.visible = recording.supported;
  support.position.fromArray(sample.rootPositionM);
  for (const [side, i] of [['left', 0], ['right', 1]]) {
    $(`${side}-force`).textContent = `${sample.footForcesN[i].toFixed(1)} N`;
    $(`${side}-contact`).textContent = sample.footContacts[i] ? '● 接地' : '○ 空中';
    $(`${side}-contact`).className = sample.footContacts[i] ? 'contact' : 'air';
  }
  $('clock').textContent = `${sample.timeSeconds.toFixed(2)} / ${recording.durationSeconds.toFixed(2)} 秒`;
  $('seek').value = time;
  $('root-height').textContent = Number.isFinite(sample.headHeightM)
    ? `頭の高さ ${sample.headHeightM.toFixed(3)} m ／ 腰 ${sample.rootPositionM[1].toFixed(3)} m`
    : `腰の高さ ${sample.rootPositionM[1].toFixed(3)} m`;
  $('body-touch').textContent = sample.bodyContactIds
    ? sample.bodyContactIds.map((id, i) => `${id}: ${sample.bodyContactForcesN[i].toFixed(1)} N`).join('\n')
    : 'この旧記録には全身の接触データがありません。';
  $('gravity-vector').textContent = `重力（身体座標）: ${sample.projectedGravity.map(v => v.toFixed(3)).join(', ')}`;
  drawCharts();
  renderer.render(scene, camera);
}

function buildColliders() {
  for (const child of [...collisionGroup.children]) {
    child.geometry.dispose(); child.material.dispose(); collisionGroup.remove(child);
  }
  for (const spec of recording.geometrySpecs) {
    const [a, b, c] = spec.size;
    let geometry;
    if (spec.type === 2) geometry = new THREE.SphereGeometry(a, 10, 8);
    else if (spec.type === 3) {
      geometry = new THREE.CapsuleGeometry(a, 2 * b, 4, 8);
      geometry.rotateX(Math.PI / 2);
    } else if (spec.type === 6) geometry = new THREE.BoxGeometry(2 * a, 2 * b, 2 * c);
    else throw Error(`Unsupported collision shape ${spec.type}`);
    collisionGroup.add(new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({
      color: spec.name.startsWith('left_') ? '#8baed0' : spec.name.startsWith('right_') ? '#c49cb8'
        : spec.name === 'head_geom' ? '#ead3b0' : '#839b9f',
      roughness: 0.65, wireframe: $('colliders').checked,
    })));
  }
}

async function loadRecording(name) {
  pending = true;
  $('toggle').disabled = true;
  try {
    const next = await json(trainingRun
      ? `/api/training/${trainingRun}/${name}` : `/runs/sensory/${name}.json`);
    if (next.bodyMapId !== body.id || next.jointIds.some((id, i) => id !== body.joints[i].id)) {
      throw Error('物理モデルとボーン対応表のバージョンが一致しません。');
    }
    if (next.jointIds.length !== body.joints.length || !next.frames.length) throw Error('Invalid recording');
    recording = next;
    if (next.task?.initialPose === 'splayed-crawl') {
      camera.position.set(1.65, 1.2, 2.8);
      controls.target.set(0, 0.22, 0);
      controls.update();
      $('brain-boundary').textContent = '頭の中心の高さを評価します。前進の加点はありません。感覚の割り当ては工学的なものです。';
    } else {
      camera.position.set(1.35, 1.1, 3.25);
      controls.target.set(0, 0.78, 0);
      controls.update();
    }
    buildColliders();
    $('mode-badge').textContent = next.ablateBrain ? '脳計算あり・関節への脳出力は遮断' : next.brainConnected
      ? `全脳接続・${next.learned ? '学習後' : next.neuralDriveEnabled ? '初期方策（脳駆動あり）' : '初期方策（脳→関節の重みゼロ）'}・支持なし`
      : next.supported ? '腰支持あり・感覚校正・未学習' : '支持なし・バランス制御未学習';
    $('seek').max = next.durationSeconds;
    $('receptor-title').textContent = `人工受容器の活動：${next.channels.length}チャネル`;
    $('receptor-legend').textContent = next.channels.length === 126
      ? '紫：関節感覚72 ／ 緑：12部位の接触48 ／ 青：重力方向6。0〜200 Hzの外部刺激。'
      : '紫：位置・速度72 ／ 緑：足裏の接触8。0〜200 Hzの外部刺激。';
    $('input-summary').textContent = `${next.channels.length}感覚チャネル → 上行性ニューロン候補への外部刺激`;
    $('scenario').value = name;
    if (trainingRun) {
      const selected = $('scenario').selectedOptions[0];
      if (name === 'best') selected.textContent = next.learned ? '学習の最良結果' : '初期方策（まだ改善なし）';
    }
    pending = false;
    showFrame(0);
    $('toggle').disabled = false;
  } catch (error) { pending = false; throw error; }
}

try {
  body = await json('/config/body-map.json');
  const binding = await json(trainingRun ? '/data/brain/compiled/input-binding.json' : '/data/brain/flywire-783-input-binding.json');
  $('input-summary').textContent = `80感覚チャネル → ${binding.neurons.length.toLocaleString()}個の上行性ニューロン候補`;
  for (const [index, joint] of body.joints.entries()) {
    const option = document.createElement('option');
    option.value = index; option.textContent = `${joint.bone} / ${joint.id.split('_').at(-1)}`;
    $('joint').append(option);
  }
  $('joint').value = 3;
  if (trainingRun) {
    $('scenario').replaceChildren();
    const availability = (await json(`/api/training/${trainingRun}/status`)).recordingsAvailable;
    for (const [value, label] of [['best', '学習の最良結果（状態1）'], ['best-second', '学習の最良結果（状態2）'], ['baseline', '初期パラメータ（状態1）'], ['baseline-second', '初期パラメータ（状態2）'], ['evaluation-replay', '別seedの検証'], ['ablation', '脳出力遮断']]) {
      const option = document.createElement('option'); option.value = value; option.textContent = label;
      option.disabled = !availability?.[value];
      $('scenario').append(option);
    }
    $('brain-boundary').textContent = '実際の接続データを使ったLIF脳からの出力で制御しています。感覚の割り当ては工学的なもので、自立歩行の獲得は別途評価が必要です。';
  }
  await loadRecording(trainingRun ? (query.get('replay') || 'best') : 'calibration');
  $('joint').onchange = () => showFrame(time);
  $('seek').oninput = event => { playing = false; $('toggle').textContent = '再生'; showFrame(Number(event.target.value)); };
  $('toggle').onclick = () => { playing = !playing; $('toggle').textContent = playing ? '一時停止' : '再生'; };
  $('colliders').onchange = event => {
    collisionGroup.children.forEach(shape => { shape.material.wireframe = event.target.checked; });
    showFrame(time);
  };
  $('scenario').onchange = event => {
    loadRecording(event.target.value).catch(error => { $('error').textContent = String(error); });
  };
  window.setSensoryTime = showFrame;
  window.loadSensoryScenario = loadRecording;
  window.getSensoryStatus = () => ({
    mode: recording.mode, supported: recording.supported, learned: recording.learned, brainConnected: recording.brainConnected,
    frameIndex: index, sample, visualModel: 'original-physics-primitives',
    modelPosition: sample.rootPositionM, modelQuaternion: sample.rootQuaternion,
    geometryPoses: collisionGroup.children.map(shape => ({ position: shape.position.toArray(), quaternion: shape.quaternion.toArray() })),
    collisionVisible: collisionGroup.visible, wireframe: $('colliders').checked,
  });
  window.sensoryReady = true;
  window.getSensoryRecording = () => recording;
  const capture = new URLSearchParams(location.search).has('capture');
  let previous = performance.now();
  if (!capture) renderer.setAnimationLoop(now => {
    const dt = Math.min(0.1, (now - previous) / 1000); previous = now;
    if (!capture && playing && !pending) {
      const next = time + dt;
      showFrame(next > recording.durationSeconds ? 0 : next);
    } else renderer.render(scene, camera);
  });
} catch (error) {
  window.sensoryError = String(error);
  $('error').textContent = String(error);
  console.error(error);
}
