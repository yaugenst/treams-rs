// Build actual browser ES modules and run their numerical qualification in Node.
// Optional: --browser /path/to/chrome also runs the identical check over HTTP in Chrome.
import {spawn, spawnSync} from "node:child_process";
import {createHash} from "node:crypto";
import {mkdtemp, readFile, rm, writeFile} from "node:fs/promises";
import {createServer} from "node:http";
import {tmpdir} from "node:os";
import {dirname, join, resolve} from "node:path";
import {fileURLToPath, pathToFileURL} from "node:url";
import {gzipSync} from "node:zlib";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
if (process.argv.length !== 2 && !(process.argv.length === 4 && process.argv[2] === "--browser")) {
  throw new Error("usage: node scripts/check_wasm.mjs [--browser /path/to/chrome]");
}
const pkg = join(root, "target/wasm-pkg");
for (const command of [
  ["cargo", "build", "--locked", "--release", "-p", "treams-wasm", "--target", "wasm32-unknown-unknown"],
  ["wasm-bindgen", "--target", "web", "--out-dir", pkg, join(root, "target/wasm32-unknown-unknown/release/treams_wasm.wasm")],
]) {
  const result = spawnSync(command[0], command.slice(1), {cwd: root, stdio: "inherit"});
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}
await writeFile(join(pkg, "package.json"), '{"type":"module"}\n');
const binary = await readFile(join(pkg, "treams_wasm_bg.wasm"));
const module = await import(pathToFileURL(join(pkg, "treams_wasm.js")));
await module.default({module_or_path: binary});
const {verify} = await import("../crates/treams-wasm/tests/verify.mjs");
const reference = JSON.parse(await readFile(join(root, "crates/treams-wasm/tests/reference.json"), "utf8"));
const report = {
  node: process.version,
  wasm_bytes: binary.length,
  wasm_gzip_bytes: gzipSync(binary, {level: 9}).length,
  wasm_sha256: createHash("sha256").update(binary).digest("hex"),
  node_results: verify(module.ScatteringSystem, reference),
};

if (process.argv[2] === "--browser") {
  if (!process.argv[3]) throw new Error("--browser requires a Chrome/Chromium executable");
  const routes = new Map([
    ["/treams_wasm.js", ["text/javascript", join(pkg, "treams_wasm.js")]],
    ["/treams_wasm_bg.wasm", ["application/wasm", join(pkg, "treams_wasm_bg.wasm")]],
    ["/verify.mjs", ["text/javascript", join(root, "crates/treams-wasm/tests/verify.mjs")]],
    ["/reference.json", ["application/json", join(root, "crates/treams-wasm/tests/reference.json")]],
  ]);
  const completed = Promise.withResolvers();
  const server = createServer(async (request, response) => {
    if (request.url === "/result" && request.method === "POST") {
      let body = "";
      for await (const chunk of request) body += chunk;
      completed.resolve(JSON.parse(body));
      response.end();
      return;
    }
    if (request.url === "/") {
      response.setHeader("Content-Type", "text/html");
      response.end(`<!doctype html><title>treams WASM qualification</title><pre id="result">Running</pre>
<script type="module">
import init, {ScatteringSystem} from './treams_wasm.js';
import {verify} from './verify.mjs';
try {
  await init();
  const reference = await (await fetch('./reference.json')).json();
  const result = verify(ScatteringSystem, reference);
  document.querySelector('#result').textContent = JSON.stringify(result);
  await fetch('/result', {method:'POST', body:JSON.stringify(result)});
} catch (e) { await fetch('/result', {method:'POST',body:JSON.stringify({passed:false,error:String(e)})}); }
</script>`);
      return;
    }
    const route = routes.get(request.url);
    if (!route) { response.writeHead(404).end(); return; }
    response.setHeader("Content-Type", route[0]);
    response.end(await readFile(route[1]));
  });
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  const profile = await mkdtemp(join(tmpdir(), "treams-wasm-chrome-"));
  const child = spawn(process.argv[3], ["--headless", "--disable-gpu", "--disable-background-networking", "--disable-component-update", "--use-mock-keychain", "--no-first-run", "--no-default-browser-check", `--user-data-dir=${profile}`, `http://127.0.0.1:${server.address().port}/`], {timeout: 60000, killSignal: "SIGKILL"});
  let stderr = "";
  child.stderr.on("data", x => {stderr += x;});
  child.on("error", completed.reject);
  const closed = new Promise(resolve => child.on("close", code => {
    completed.reject(new Error(`browser exited before qualification (code ${code}): ${stderr}`));
    resolve();
  }));
  try {
    report.browser_results = await completed.promise;
    if (!report.browser_results.passed) throw new Error(JSON.stringify(report.browser_results));
    report.browser = spawnSync(process.argv[3], ["--version"], {encoding: "utf8"}).stdout.trim();
  } finally {
    child.kill("SIGTERM");
    await closed;
    await new Promise(resolve => server.close(resolve));
    await rm(profile, {recursive: true, force: true});
  }
}
await writeFile(join(pkg, "qualification.json"), JSON.stringify(report, null, 2) + "\n");
console.log(JSON.stringify(report, null, 2));
