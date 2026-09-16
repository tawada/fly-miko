const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const api = require('../scripts/training_api.cjs');

test('snapshot reads recover from a transient rename gap or access denial', async t => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'fly-api-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const directory = path.join(root, 'runs/training/random-head-bias-02');
  fs.mkdirSync(directory, { recursive: true });
  fs.writeFileSync(path.join(directory, 'progress.json'), '{"generation":2}');
  const original = fs.promises.readFile;
  let attempts = 0;
  t.mock.method(fs.promises, 'readFile', async (...args) => {
    if (args[0].endsWith('progress.json') && ++attempts < 3) {
      throw Object.assign(Error('temporarily unavailable'), { code: attempts === 1 ? 'ENOENT' : 'EACCES' });
    }
    return original.apply(fs.promises, args);
  });
  let status, body;
  const response = { writeHead(code) { status = code; }, end(text) { body = JSON.parse(text); } };
  assert.equal(await api({ method: 'GET' }, response, root, '/api/training/random-head-bias-02/status'), true);
  assert.equal(status, 200);
  assert.equal(body.progress.generation, 2);
  assert.equal(attempts, 3);
  assert.equal(body.config, null);
  assert.deepEqual(body.history, []);
});

test('async API still delegates static files and rejects writes', async () => {
  assert.equal(await api({ method: 'GET' }, {}, '/tmp', '/demo/training.html'), false);
  let status;
  const response = { writeHead(code) { status = code; }, end() {} };
  assert.equal(await api({ method: 'POST' }, response, '/tmp', '/api/training/start'), true);
  assert.equal(status, 405);
});

test('activity API rejects stale data after the source replay changes', async t => {
  const { createHash } = require('node:crypto');
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'fly-activity-api-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const directory = path.join(root, 'runs/training/random-head-bias-02');
  fs.mkdirSync(directory, { recursive: true });
  const source = '{"frames":[]}';
  fs.writeFileSync(path.join(directory, 'best.json'), source);
  fs.writeFileSync(path.join(directory, 'best-activity.json'), JSON.stringify({
    sourceSha256: createHash('sha256').update(source).digest('hex'), frames: [],
  }));
  let status;
  const response = { writeHead(code) { status = code; }, end() {} };
  await api({ method: 'GET' }, response, root, '/api/training/random-head-bias-02/best-activity');
  assert.equal(status, 200);
  fs.writeFileSync(path.join(directory, 'best.json'), '{"frames":[1]}');
  await api({ method: 'GET' }, response, root, '/api/training/random-head-bias-02/best-activity');
  assert.equal(status, 409);
});

test('only the current run is listed and legacy endpoints stay private', async t => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'fly-current-api-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  for (const name of ['random-head-bias-02', 'random-head-bias-01']) {
    const directory = path.join(root, 'runs/training', name);
    fs.mkdirSync(directory, { recursive: true });
    fs.writeFileSync(path.join(directory, 'config.json'), JSON.stringify({ name, createdAt: '2026-09-16' }));
    fs.writeFileSync(path.join(directory, 'progress.json'), '{"status":"training"}');
  }
  let status, value;
  const response = { writeHead(code) { status = code; }, end(text) { value = JSON.parse(text); } };
  await api({ method: 'GET' }, response, root, '/api/training/runs');
  assert.equal(status, 200);
  assert.deepEqual(value.map(run => run.name), ['random-head-bias-02']);
  await api({ method: 'GET' }, response, root, '/api/training/random-head-bias-01/status');
  assert.equal(status, 404);
});
