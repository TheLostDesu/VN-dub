// Writes dist/build-stamp.json after `vite build`: a hash of the sources the bundle is made of, so the
// app and the tests can tell a stale build (vndub_app/webbuild.py computes the same hash).
import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const web = resolve(dirname(fileURLToPath(import.meta.url)), "..");
// keep in sync with SOURCES in vndub_app/webbuild.py
const SOURCES = ["src", "index.html", "package.json", "package-lock.json", "vite.config.ts", "tsconfig.json", "../../app/i18n/ru.json", "../../app/i18n/en.json"];

function walk(dir, rel, out) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p, `${rel}/${name}`, out);
    else out.push([`${rel}/${name}`, p]);
  }
}

const files = [];
for (const rel of SOURCES) {
  const p = resolve(web, rel);
  if (!existsSync(p)) continue;
  if (statSync(p).isDirectory()) walk(p, rel, files);
  else files.push([rel, p]);
}
// Python sorts the (relpath, path) tuples by relpath with str comparison: the same code-point order
files.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));

const h = createHash("sha256");
for (const [rel, p] of files) {
  const bytes = readFileSync(p);
  const lf = Buffer.from(bytes.toString("latin1").replace(/\r\n/g, "\n"), "latin1");
  h.update(`${rel}\0${createHash("sha256").update(lf).digest("hex")}\n`);
}
const stamp = { sources_sha256: h.digest("hex"), files: files.length, built_at: new Date().toISOString() };
writeFileSync(join(web, "dist", "build-stamp.json"), JSON.stringify(stamp, null, 2) + "\n");
console.log(`build stamp: ${stamp.sources_sha256.slice(0, 12)} (${stamp.files} files)`);
