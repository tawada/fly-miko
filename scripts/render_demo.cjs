const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const { once } = require('node:events');
const os = require('node:os');
const { chromium } = require('playwright');
if (process.argv.length === 2) process.argv.push('--render-training');

const root = path.resolve(__dirname, '..');
const output = path.join(root, 'renders');
fs.mkdirSync(output, { recursive: true });
const fontConfig = path.join(output, 'fonts.conf');
const xmlEscape = value => value.replaceAll('&', '&amp;').replaceAll('<', '&lt;');
fs.writeFileSync(fontConfig, `<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd">
<fontconfig><dir>${xmlEscape(path.join(root, 'assets/fonts'))}</dir>
<dir>/usr/share/fonts</dir><cachedir>${xmlEscape(path.join(output, '.font-cache'))}</cachedir></fontconfig>`);
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.png': 'image/png', '.json': 'application/json' };
// Serve only files needed by the preview; .env and other workspace files stay private.
const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://localhost');
    if (await require('./training_api.cjs')(req, res, root, url.pathname)) return;
    const pathname = decodeURIComponent(url.pathname === '/' ? '/demo/index.html' : url.pathname);
    const target = path.resolve(root, `.${pathname}`);
    const allowed = ['demo', 'src/bridge', 'config', 'assets/fonts', 'node_modules/three'].some(directory => {
      const base = path.join(root, directory) + path.sep;
      return target.startsWith(base);
    }) || [
      'data/brain/flywire-783-binding.json', 'data/brain/flywire-783-input-binding.json',
      'data/brain/compiled/input-binding.json', 'data/brain/compiled/output-binding.json',
    ].some(file => target === path.join(root, file));
    if (!allowed || !fs.existsSync(target) || !fs.statSync(target).isFile() || fs.realpathSync(target) !== target) {
      res.writeHead(404).end(); return;
    }
    res.setHeader('Content-Type', path.extname(target) === '.mjs' ? 'text/javascript' : mime[path.extname(target)] || 'application/octet-stream');
    fs.createReadStream(target).pipe(res);
  } catch {
    res.writeHead(400).end();
  }
});

(async () => {
  const preview = process.argv.includes('--serve');
  server.listen(preview ? 8000 : 0, preview ? '0.0.0.0' : '127.0.0.1');
  await once(server, 'listening');
  const address = `http://127.0.0.1:${server.address().port}/`;
  if (process.argv.includes('--serve')) {
    console.log(`Preview: ${address}`);
    console.log('Listening on 0.0.0.0:8000 (container network access enabled)');
    return;
  }
  let browser;
  try {
    browser = await chromium.launch({
      headless: true,
      env: {
        ...process.env,
        FONTCONFIG_FILE: fontConfig,
        LD_LIBRARY_PATH: [
          process.env.LD_LIBRARY_PATH,
          path.join(os.homedir(), '.cache/fly-miko-browser-libs/usr/lib/x86_64-linux-gnu'),
          path.join(os.homedir(), '.cache/fly-miko-browser-libs/lib/x86_64-linux-gnu'),
        ].filter(Boolean).join(':'),
      },
      args: ['--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
    });
    const trainingVideo = process.argv.includes('--render-training');
    const trainingMode = trainingVideo || process.argv.includes('--check-training');
    const page = await browser.newPage({
      viewport: trainingVideo ? { width: 1920, height: 1080 }
        : { width: 1440, height: 1030 },
      deviceScaleFactor: 1,
    });
    page.on('pageerror', error => console.error(error));
    if (trainingVideo) {
      await require('./render_training.cjs')(page, address, root, output);
      return;
    }
    if (trainingMode) {
      await require('./check_training_page.cjs')(page, address, root, output);
      return;
    }
    throw Error('Use --serve, --render-training or --check-training');
  } finally {
    if (browser) await browser.close();
    server.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
  server.close();
});
