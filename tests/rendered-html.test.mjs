import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server renders Caro Lab metadata without starter markers", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /<title>Caro Lab — Self-Learning Chess AI<\/title>/i);
  assert.match(html, /og:image/);
  assert.match(html, /og\.png/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|react-loading-skeleton/i);
});

test("dashboard source contains the required authoritative product surfaces", async () => {
  const [page, layout, css, packageJson] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);
  for (const label of ["Overview", "Live Arena", "Game History", "Training Analytics", "Checkpoints", "Model Battle", "Configuration", "System Logs"]) {
    assert.match(page, new RegExp(label));
  }
  assert.match(page, /\/api\/training\/start/);
  assert.match(page, /\/ws\/live/);
  assert.match(page, /ChessBoard/);
  assert.match(layout, /Caro Lab — Self-Learning Chess AI/);
  assert.match(css, /--lime:#b8f25f/);
  assert.match(css, /aspect-ratio:1\/1/);
  assert.match(css, /grid-template:repeat\(8,minmax\(0,1fr\)\)/);
  assert.match(page, /MODEL PROMOTION/);
  assert.match(page, /candidate_progress/);
  assert.match(page, /stop_after_promotion/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton|recharts/);
});
