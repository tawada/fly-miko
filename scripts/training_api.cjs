const fs = require('node:fs');
const path = require('node:path');
const { setTimeout: delay } = require('node:timers/promises');
const { createHash } = require('node:crypto');

module.exports = async function trainingApi(req, res, root, pathname) {
  if (!pathname.startsWith('/api/training/')) return false;
  const send = (status, value) => {
    res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
    res.end(JSON.stringify(value));
  };
  if (req.method !== 'GET') { send(405, { error: 'Read-only training API' }); return true; }
  const runs = path.join(root, 'runs/training');
  const valid = name => /^[A-Za-z0-9_-]{1,80}$/.test(name);
  const read = async (directory, file, fallback = null) => {
    const target = path.join(directory, file);
    for (let attempt = 0; ; attempt++) {
      try {
        const real = await fs.promises.realpath(target);
        if (!real.startsWith(fs.realpathSync(runs) + path.sep)) throw Error('Invalid run file');
        return JSON.parse(await fs.promises.readFile(real, 'utf8'));
      } catch (error) {
        if (attempt < 4 && ['ENOENT', 'EACCES', 'EPERM', 'EBUSY'].includes(error.code)) {
          await delay(10 * 2 ** attempt);
          continue;
        }
        if (error.code === 'ENOENT') return fallback;
        throw error;
      }
    }
  };
  try {
    if (pathname === '/api/training/runs') {
      const entries = fs.existsSync(runs) ? (await Promise.all(fs.readdirSync(runs, { withFileTypes: true })
        .filter(entry => entry.isDirectory() && entry.name === 'random-head-bias-02')
        .map(async entry => {
          const directory = path.join(runs, entry.name);
          const [config, progress] = await Promise.all([read(directory, 'config.json'), read(directory, 'progress.json')]);
          return { name: entry.name, config, progress };
        }))).filter(entry => entry.config).sort((a, b) => b.config.createdAt.localeCompare(a.config.createdAt)) : [];
      send(200, entries);
      return true;
    }
    const match = pathname.match(/^\/api\/training\/([A-Za-z0-9_-]{1,80})\/(status|baseline|best|best-second|baseline-second|evaluation-replay|ablation|decoder|(?:baseline|best|best-second|baseline-second|evaluation-replay|ablation)-activity)$/);
    if (!match || match[1] !== 'random-head-bias-02' || !fs.existsSync(path.join(runs, match[1]))) { send(404, { error: 'Unknown run' }); return true; }
    const directory = path.join(runs, match[1]);
    if (match[2] === 'status') {
      const [config, progress, history, evaluation, baseline] = await Promise.all([
        read(directory, 'config.json'), read(directory, 'progress.json'),
        read(directory, 'history.json', []), read(directory, 'evaluation.json', []),
        read(directory, 'baseline-metrics.json'),
      ]);
      send(200, { config, progress, history, evaluation, baseline,
        recordingsAvailable: Object.fromEntries(['baseline', 'best', 'best-second', 'baseline-second', 'evaluation-replay', 'ablation']
          .map(name => [name, fs.existsSync(path.join(directory, `${name}.json`))])) });
    } else if (match[2].endsWith('-activity')) {
      const data = await read(directory, `${match[2]}.json`);
      if (!data) { send(404, { error: 'Activity not recorded' }); return true; }
      const sourcePath = await fs.promises.realpath(path.join(directory, `${match[2].slice(0, -9)}.json`));
      if (!sourcePath.startsWith(fs.realpathSync(runs) + path.sep)) throw Error('Invalid run file');
      const source = await fs.promises.readFile(sourcePath);
      if (createHash('sha256').update(source).digest('hex') !== data.sourceSha256) {
        send(409, { error: 'Activity belongs to an older recording' }); return true;
      }
      send(200, data);
    } else {
      const data = await read(directory, `${match[2]}.json`);
      send(data ? 200 : 404, data || { error: 'Recording not available yet' });
    }
  } catch (error) {
    send(500, { error: 'Cannot read training snapshot' });
  }
  return true;
};
