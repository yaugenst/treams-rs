import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { cp, mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";

async function writeNotices() {
  const cargo = (...args) =>
    execFileSync("cargo", args, { cwd: "..", encoding: "utf8" });
  const { packages } = JSON.parse(
    cargo(
      "metadata",
      "--locked",
      "--filter-platform",
      "wasm32-unknown-unknown",
      "--format-version",
      "1",
    ),
  );
  const selected = new Set(
    cargo(
      "tree",
      "--offline",
      "--locked",
      "-p",
      "treams-wasm",
      "--target",
      "wasm32-unknown-unknown",
      "--edges",
      "normal",
      "--prefix",
      "none",
      "--format",
      "{p}",
    )
      .trim()
      .split("\n")
      .map((line) => {
        const [, name, version] = line.match(/^(\S+) v(\S+)/);
        return `${name}@${version}`;
      }),
  );
  const supplements = JSON.parse(
    await readFile("dependency-licenses.json", "utf8"),
  );
  const sections = [
    "Light Lab — licenses and third-party notices",
    "Includes the selected wasm32 Cargo normal-dependency graph, conservatively including procedural macros.\nRust standard-library notices accompany this file in RUST-STDLIB-NOTICES.html.",
  ];
  for (const file of [
    "LICENSE",
    "LICENSE.treams",
    "LICENSE.xsf",
    "THIRD_PARTY_NOTICES.md",
  ]) {
    sections.push(`${file}\n\n${await readFile(`../${file}`, "utf8")}`);
  }
  for (const pkg of packages
    .filter((pkg) => pkg.source && selected.has(`${pkg.name}@${pkg.version}`))
    .sort((a, b) =>
      `${a.name}@${a.version}`.localeCompare(`${b.name}@${b.version}`, "en"),
    )) {
    const key = `${pkg.name}@${pkg.version}`;
    const directory = dirname(pkg.manifest_path);
    const files = new Set(
      (await readdir(directory, { withFileTypes: true }))
        .filter(
          (file) =>
            file.isFile() &&
            /^(licen[cs]e|copying|notice|copyright)([._-]|$)/i.test(file.name),
        )
        .map((file) => file.name),
    );
    if (pkg.license_file) files.add(pkg.license_file);
    sections.push(
      `${key}\nLicense: ${pkg.license}\nSource: ${pkg.repository || pkg.source}`,
    );
    for (const file of [...files].sort()) {
      sections.push(
        `${key} — ${basename(file)}\n\n${await readFile(resolve(directory, file), "utf8")}`,
      );
    }
    if (!files.size) {
      const supplement = supplements.find((item) =>
        item.packages.includes(key),
      );
      if (
        !supplement ||
        createHash("sha256").update(supplement.text).digest("hex") !==
          supplement.sha256
      ) {
        throw new Error(`Missing or invalid license text for ${key}`);
      }
      sections.push(
        `${key} — repository license\nSource: ${supplement.source}\n\n${supplement.text}`,
      );
    }
  }
  const toolchain = execFileSync("rustc", ["--print", "sysroot"], {
    cwd: "..",
    encoding: "utf8",
  }).trim();
  await cp(
    join(toolchain, "share/doc/rust/COPYRIGHT-library.html"),
    "dist/RUST-STDLIB-NOTICES.html",
  );
  await writeFile(
    "dist/NOTICES.txt",
    `${sections.join("\n\n" + "=".repeat(72) + "\n\n")}\n`,
  );
}

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
  for (const file of [
    "index.html",
    "styles.css",
    "advanced.html",
    "advanced.css",
  ])
    await cp(file, `dist/${file}`);
  await cp("wasm", "dist/wasm", { recursive: true });
  await writeNotices();
}
