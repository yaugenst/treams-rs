import { cp, mkdir } from "node:fs/promises";
if (process.argv.includes("--bindings")) {
  await mkdir("wasm", { recursive: true });
  for (const file of [
    "treams_wasm.js",
    "treams_wasm_bg.wasm",
    "treams_wasm.d.ts",
    "treams_wasm_bg.wasm.d.ts",
  ]) {
    await cp(`../target/wasm-pkg/${file}`, `wasm/${file}`);
  }
} else {
  await mkdir("dist", { recursive: true });
  for (const file of ["index.html", "styles.css"])
    await cp(file, `dist/${file}`);
  await cp("wasm", "dist/wasm", { recursive: true });
}
