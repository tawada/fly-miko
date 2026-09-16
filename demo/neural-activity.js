const $ = id => document.getElementById(id);
const colors = ['#151b2b', '#9874ec', '#ef8aba', '#ffe29a'];
const role = bits => bits & 2 ? 2 : bits & 1 ? 0 : 1;
const roleNames = ['入力候補', '内部', '出力候補'];
function chart(id) {
  const canvas = $(id);
  canvas.width = Math.max(1, canvas.clientWidth); canvas.height = canvas.clientHeight;
  return { ctx: canvas.getContext('2d'), w: canvas.width, h: canvas.height };
}
export class ActivityView {
  constructor() {
    this.token = 0; this.selected = null; this.time = 0;
    $('activity-role').onchange = () => { this.buildRows(); this.draw(); };
    $('neuron-find').onclick = () => {
      if (!this.data) return;
      const index = this.data.neuronIds.indexOf($('neuron-search').value.trim());
      if (index < 0) { $('neuron-detail').textContent = 'この記録にそのIDはありません。'; return; }
      this.selected = index; this.buildRows(); this.draw();
    };
    $('neuron-search').onkeydown = event => { if (event.key === 'Enter') $('neuron-find').click(); };
    $('neuron-map').onclick = event => {
      if (!this.data) return;
      const canvas = $('neuron-map'), rect = canvas.getBoundingClientRect();
      const index = Math.floor((event.clientY - rect.top) / rect.height * canvas.height) * canvas.width
        + Math.floor((event.clientX - rect.left) / rect.width * canvas.width);
      if (index >= this.data.neuronIds.length) return;
      this.selected = index; $('neuron-search').value = this.data.neuronIds[index];
      this.buildRows(); this.draw();
    };
    addEventListener('resize', () => this.draw());
  }
  accepts(index) {
    const filter = $('activity-role').value, bits = this.data.roles[index];
    return filter === 'all' || (filter === 'input' ? bits & 1 : filter === 'output' ? bits & 2 : bits === 0);
  }
  async load(run, scenario, recording) {
    const token = ++this.token;
    this.data = null; this.selected = null;
    window.neuralActivityStatus = null;
    for (const id of ['neuron-map', 'activity-trend', 'neuron-raster']) {
      const c = $(id); c.getContext('2d').clearRect(0, 0, c.width, c.height);
    }
    $('activity-stats').textContent = ''; $('neuron-detail').textContent = ''; $('activity-time').textContent = '';
    $('activity-message').textContent = '全脳の発火記録を読み込み中…';
    try {
      const response = await fetch(`/api/training/${encodeURIComponent(run)}/${scenario}-activity`, { cache: 'no-store' });
      if (token !== this.token) return;
      if (!response.ok) {
        if (![404, 409].includes(response.status)) throw Error(`HTTP ${response.status}`);
        $('activity-message').textContent = `${response.status === 409 ? '発火記録が以前の試行のものです。' : 'この試行には全脳の個別発火記録がありません。'} 作成: npm run brain:record-activity -- --name ${run} --scenario ${scenario}（作成後に「最新結果を読込」）`;
        return;
      }
      const data = await response.json();
      if (token !== this.token) return;
      if (data.policyFingerprint !== recording.policyFingerprint || data.seed !== recording.metrics.seed
        || data.frames.length !== recording.frames.length || data.neuronIds.length !== recording.brainModel.neuronCount
        || data.roles.length !== data.neuronIds.length) throw Error('再生と発火記録が一致しません');
      // Guard against a best.json update racing the iframe and activity requests.
      data.frames.forEach((frame, i) => {
        if (Math.abs(frame.timeSeconds - recording.frames[i].timeSeconds) > 1e-8
          || frame.counts.reduce((sum, [, n]) => sum + n, 0) !== (recording.frames[i].brain?.spikeCount ?? 0)) {
          throw Error('再生と発火記録の時刻・発火数が一致しません');
        }
      });
      this.data = data;
      this.totals = new Float64Array(data.neuronIds.length);
      this.groups = data.frames.map(frame => {
        const counts = [0, 0, 0];
        for (const [index, count] of frame.counts) { this.totals[index] += count; counts[role(data.roles[index])] += count; }
        return counts;
      });
      this.buildRows();
      $('activity-message').textContent = `${data.neuronIds.length.toLocaleString()}ニューロンの計算上の発火。保存済み感覚入力から再計算し、全区間の発火総数と出力発火を元の記録と照合済み。`;
      this.draw();
    } catch (error) {
      if (token === this.token) { this.data = null; $('activity-message').textContent = `発火表示を読み込めません: ${error.message}`; }
    }
  }
  update(run, scenario, recording, time) {
    if (this.recording !== recording || this.run !== run || this.scenario !== scenario) {
      this.recording = recording; this.run = run; this.scenario = scenario; this.time = time;
      this.load(run, scenario, recording); return;
    }
    if (this.time !== time) { this.time = time; this.draw(); }
  }
  buildRows() {
    if (!this.data) return;
    this.mask = Uint8Array.from(this.data.roles, (_, i) => this.accepts(i) ? 1 : 0);
    this.rows = [...this.totals.keys()].filter(i => this.accepts(i) && this.totals[i] > 0)
      .sort((a, b) => this.totals[b] - this.totals[a]).slice(0, 24);
    if (this.selected !== null) this.rows = [this.selected, ...this.rows.filter(i => i !== this.selected)].slice(0, 24);
    this.raster = null;
  }
  draw() {
    if (!this.data) return;
    const data = this.data, frameIndex = Math.min(data.frames.length - 1, Math.max(0, Math.round(this.time / data.binSeconds)));
    const frame = data.frames[frameIndex], counts = new Map(frame.counts);
    const canvas = $('neuron-map');
    canvas.height = Math.ceil(data.neuronIds.length / canvas.width);
    const ctx = canvas.getContext('2d'), image = ctx.createImageData(canvas.width, canvas.height);
    const palette = [[21, 27, 43], [152, 116, 236], [239, 138, 186], [255, 226, 154]];
    let active = 0, spikes = 0, visible = 0;
    for (let i = 0; i < data.neuronIds.length; i++) {
      const accepted = this.mask[i], count = counts.get(i) || 0;
      if (accepted) { visible++; if (count) active++; spikes += count; }
      const rgb = accepted ? palette[Math.min(3, count)] : [53, 57, 65];
      const offset = i * 4;
      image.data[offset] = rgb[0]; image.data[offset + 1] = rgb[1];
      image.data[offset + 2] = rgb[2]; image.data[offset + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
    if (this.selected !== null) {
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1;
      ctx.strokeRect(this.selected % canvas.width - 2.5, Math.floor(this.selected / canvas.width) - 2.5, 6, 6);
    }
    $('activity-time').textContent = `${frame.timeSeconds.toFixed(2)} s`;
    $('activity-stats').textContent = `この20ms：${active.toLocaleString()} / ${visible.toLocaleString()}個が発火 · ${spikes.toLocaleString()}スパイク`;
    if (this.selected !== null) {
      const i = this.selected, bits = data.roles[i];
      $('neuron-detail').textContent = `ID ${data.neuronIds[i]} · ${bits === 3 ? '入力・出力候補' : roleNames[role(bits)]} · この区間 ${counts.get(i) || 0}回 · 試行全体 ${this.totals[i]}回 · 平均 ${(this.totals[i] / this.recording.durationSeconds).toFixed(2)} Hz`;
    } else $('neuron-detail').textContent = 'マップをクリックするかIDを入力すると、個別の発火回数を表示します。';
    this.drawTrend(frame.timeSeconds);
    this.drawRaster(frame.timeSeconds);
    window.neuralActivityStatus = { scenario: this.scenario, timeSeconds: frame.timeSeconds,
      neuronCount: data.neuronIds.length, activeCount: active, spikeCount: spikes, selected: this.selected };
  }
  drawTrend(time) {
    const { ctx, w, h } = chart('activity-trend'), left = 44, duration = this.recording.durationSeconds;
    const max = Math.max(1, ...this.groups.map(row => Math.max(...row)));
    ctx.font = '10px sans-serif'; ctx.fillStyle = '#87798a'; ctx.fillText(String(max), 0, 12); ctx.fillText('0', 0, h - 16);
    ['#9874ec', '#638cd5', '#e480a8'].forEach((color, group) => {
      ctx.strokeStyle = color; ctx.beginPath();
      this.groups.forEach((counts, i) => {
        const x = left + this.data.frames[i].timeSeconds / duration * (w - left - 5);
        const y = h - 20 - counts[group] / max * (h - 32);
        if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      }); ctx.stroke();
    });
    this.axis(ctx, w, h, left, time);
  }
  axis(ctx, w, h, left, time) {
    const duration = this.recording.durationSeconds;
    ctx.fillStyle = '#87798a'; ctx.font = '10px sans-serif';
    for (let i = 0; i <= 4; i++) ctx.fillText(`${(duration * i / 4).toFixed(1)}s`, left + i / 4 * (w - left - 30), h - 2);
    const x = left + time / duration * (w - left - 5);
    ctx.strokeStyle = '#b66292'; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h - 15); ctx.stroke();
  }
  drawRaster(time) {
    const { ctx, w, h } = chart('neuron-raster'), left = 140;
    if (!this.raster || this.raster.width !== w || this.raster.height !== h) {
      this.raster = document.createElement('canvas'); this.raster.width = w; this.raster.height = h;
      const c = this.raster.getContext('2d'), rowHeight = (h - 20) / Math.max(1, this.rows.length);
      c.font = '9px monospace'; c.fillStyle = '#87798a';
      this.rows.forEach((index, row) => c.fillText(this.data.neuronIds[index], 0, (row + .85) * rowHeight));
      const indices = new Map(this.rows.map((index, row) => [index, row]));
      for (const frame of this.data.frames) for (const [index, count] of frame.counts) {
        if (!indices.has(index)) continue;
        c.fillStyle = colors[Math.min(3, count)];
        c.fillRect(left + (frame.timeSeconds - this.data.binSeconds) / this.recording.durationSeconds * (w - left - 5),
          indices.get(index) * rowHeight, Math.max(1, this.data.binSeconds / this.recording.durationSeconds * (w - left - 5)), Math.max(1, rowHeight - 1));
      }
    }
    ctx.drawImage(this.raster, 0, 0); this.axis(ctx, w, h, left, time);
  }
}
