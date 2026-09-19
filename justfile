set shell := ["bash", "-euo", "pipefail", "-c"]

# Keep local build paths out of distributed binaries, including dependency panics.
export CARGO_ENCODED_RUSTFLAGS := ```
    python3 - <<'PY'
    import os
    from pathlib import Path
    flags = os.environ.get("CARGO_ENCODED_RUSTFLAGS", "\x1f".join(os.environ.get("RUSTFLAGS", "").split())).split("\x1f")
    for path in (Path.home(), os.environ.get("CARGO_HOME"), os.environ.get("RUSTUP_HOME"), Path.cwd()):
        if path:
            flags.append(f"--remap-path-prefix={Path(path).resolve()}=/build")
    print("\x1f".join(filter(None, flags)))
    PY
    ```

build-ext:
    uv run --no-sync maturin develop -m crates/treams-py/Cargo.toml

build-ext-release:
    uv run --no-sync maturin develop --release -m crates/treams-py/Cargo.toml

rust-fmt-check:
    cargo fmt --check --all

rust-lint:
    PYO3_PYTHON="$PWD/.venv/bin/python" cargo clippy --locked --workspace --all-targets -- -D warnings

# Dynamic CUDA loading can be compiled without a toolkit or a GPU.
rust-cuda-check:
    PYO3_PYTHON="$PWD/.venv/bin/python" cargo clippy --locked -p treams-py --all-targets --features cuda -- -D warnings

rust-test:
    PYO3_PYTHON="$PWD/.venv/bin/python" cargo test -p treams-core

py-format-check:
    uv run --no-sync ruff format --check .

py-lint:
    uv run --no-sync ruff check .

py-types:
    uv run --no-sync pyrefly check --summarize-errors

dependency-lock-check:
    uv lock --check
    cargo metadata --locked --format-version 1 --no-deps > /dev/null

file-hygiene:
    uv run --no-sync pre-commit run --all-files --hook-stage manual

check: file-hygiene dependency-lock-check rust-fmt-check rust-lint py-format-check py-lint py-types docs-check

docs:
    uv run --no-sync python scripts/generate_agent_docs.py

docs-check:
    uv run --no-sync python scripts/generate_agent_docs.py --check

test-py: build-ext
    uv run --no-sync pytest

test: rust-test test-py

verify: check test

ci: verify
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps

# Opt-in hardware lane: requires an NVIDIA GPU and the CUDA 13.3 cuTile toolkit.
gpu-check:
    PYO3_PYTHON="$PWD/.venv/bin/python" cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
    uv run --no-sync maturin develop --release --features cuda-tile
    cargo test --locked -p treams-cuda --features cuda --release -- --ignored
    cargo test --locked -p treams-cuda-tile --features cuda-tile --release -- --ignored
    TREAMS_TEST_CUDA=1 TREAMS_TEST_CUDA_TILE=1 uv run --no-sync pytest tests/test_cuda.py

build-wheel:
    uv run --no-sync maturin build --release --locked --out dist

check-wheel: build-wheel
    #!/usr/bin/env bash
    set -euo pipefail
    uv run --no-sync python - <<'PY'
    from pathlib import Path
    import subprocess
    import tempfile
    from zipfile import ZipFile

    wheel = max(Path("dist").glob("*.whl"), key=lambda p: p.stat().st_mtime).resolve()
    with ZipFile(wheel) as archive:
        for name in archive.namelist():
            assert str(Path.home()).encode() not in archive.read(name), f"Local home directory in wheel: {name}"
    with tempfile.TemporaryDirectory(prefix="treams-wheel-") as env:
        python = str(Path(env) / "bin/python")
        subprocess.run(["uv", "venv", "--python", ".venv/bin/python", env], check=True)
        subprocess.run(["uv", "pip", "install", "--python", python, str(wheel), "advect"], check=True)
        subprocess.run([python, "scripts/check_wheel.py"], check=True)
        subprocess.run(["uv", "pip", "install", "--python", python, f"{wheel}[io]"], check=True)
        subprocess.run([python, "-c", 'import h5py\nimport numpy as np\nimport treams_rs as tr\nfrom treams_rs import io\nsphere = tr.TMatrix.sphere(1, 1.3, 0.2, [3, (1.3, 1.1, 0.08)])\ncluster = tr.TMatrix.cluster([sphere, sphere], [[0, 0, 0], [0.7, 0.2, 0.1]])\nwith h5py.File("memory.h5", "w", driver="core", backing_store=False) as handle:\n    io.save_hdf5(handle, cluster)\n    loaded = io.load_hdf5(handle, lunit="um")\n    np.testing.assert_allclose(loaded.array, cluster.array)\n    np.testing.assert_allclose(loaded.basis.positions, cluster.basis.positions * 1e-3)\n    np.testing.assert_allclose(loaded.k0, cluster.k0 * 1e3)\n    assert loaded.material == cluster.material\nprint("Clean wheel: optional HDF5 chirality, origins and units round trip passed")'], check=True)
    PY

# Run on an idle performance host; hosted CI machines have uncontrolled CPU sharing.
bench-performance: build-ext-release
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p benchmarks/results
    for order in 3 4; do
        uv run --no-sync python scripts/benchmark_cluster.py --workload ebcm --particles 1 --lmax "$order" --samples 96 --threads 4 --require-speedup 1 --require-rss-ratio 1 > "benchmarks/results/ebcm-l${order}-q96.json"
    done
    for order in 128 512; do
        record_gates=(--require-speedup 1)
        if [[ "$order" == 128 ]]; then record_gates+=(--require-rss-ratio 1); fi
        for columns in 1 8; do
            uv run --no-sync python scripts/benchmark_cluster.py --workload internal-field-forward --particles 1 --lmax "$order" --samples "$columns" --threads 4 --require-speedup 1 --require-rss-ratio 1 > "benchmarks/results/internal-forward-l${order}-p${columns}.json"
            uv run --no-sync python scripts/benchmark_cluster.py --workload internal-field --particles 1 --lmax "$order" --samples "$columns" --threads 4 "${record_gates[@]}" > "benchmarks/results/internal-adjoint-l${order}-p${columns}.json"
        done
    done
    for order in 64 512; do
        uv run --no-sync python scripts/benchmark_cluster.py --workload plane-permutation --particles 1 --lmax "$order" --samples 1 --threads 4 --require-speedup 1 --require-rss-ratio 1 > "benchmarks/results/plane-permutation-l${order}.json"
    done
    for order in 64 512; do
        uv run --no-sync python scripts/benchmark_cluster.py --workload oriented-chirality --particles 1 --lmax "$order" --threads 4 --require-speedup 1 --require-rss-ratio 1 > "benchmarks/results/oriented-chirality-l${order}.json"
    done
    for workload in particle-cluster particle-cluster-public cylindrical-particle-cluster cylindrical-particle-cluster-public; do
        for particles in 4 16; do
            uv run --no-sync python scripts/benchmark_cluster.py --workload "$workload" --particles "$particles" --lmax 3 --threads 4 --require-speedup 1 --require-rss-ratio 1 > "benchmarks/results/${workload}-n${particles}-l3.json"
        done
    done
    for workload in bessel bessel-derivative bessel-forward bessel-derivative-forward; do
        for samples in 1 128 4096; do
            uv run --no-sync python scripts/benchmark_cluster.py --workload "$workload" --samples "$samples" --particles 1 --lmax 3 --threads 4 --require-speedup 1 --require-rss-ratio 1 > "benchmarks/results/ufunc-${workload}-n${samples}.json"
        done
    done
    for workload in angular-legendre angular-pi angular-tau angular-legendre-forward angular-pi-forward angular-tau-forward; do
        for samples in 1 128 4096; do
            uv run --no-sync python scripts/benchmark_cluster.py --workload "$workload" --samples "$samples" --particles 1 --lmax 6 --threads 4 --require-speedup 1 --require-rss-ratio 1 > "benchmarks/results/${workload}-n${samples}.json"
        done
    done
    for workload in wigner wigner-forward wigner-small-forward wigner3j-forward incgamma-forward intkambe-forward; do
        for samples in 1 128 4096; do
            uv run --no-sync python scripts/benchmark_cluster.py --workload "$workload" --samples "$samples" --particles 1 --lmax 6 --threads 4 --require-speedup 1 --require-rss-ratio 1 > "benchmarks/results/${workload}-n${samples}.json"
        done
    done
    for workload in cylindrical-field cylindrical-field-axial; do
        for samples in 128 2048; do
            uv run --no-sync python scripts/benchmark_cluster.py --workload "$workload" --samples "$samples" --particles 4 --lmax 3 --threads 4 --require-speedup 1 --require-rss-ratio 1 > "benchmarks/results/${workload}-s${samples}.json"
        done
    done
    for workload in cylindrical-expansion cylindrical-expansion-axial cylindrical-periodic cylindrical-periodic-axial; do
        for particles in 4 16; do
            uv run --no-sync python scripts/benchmark_cluster.py --workload "$workload" --particles "$particles" --lmax 3 --threads 4 --require-speedup 1 --require-rss-ratio 1 > "benchmarks/results/${workload}-n${particles}-l3.json"
        done
    done
    for name in car2cyl car2sph cyl2car cyl2sph sph2car sph2cyl car2pol pol2car vcar2cyl vcar2sph vcyl2car vcyl2sph vsph2car vsph2cyl vcar2pol vpol2car; do
        for samples in 1 128 65536; do
            uv run --no-sync python scripts/benchmark_cluster.py --workload "coordinate-${name}-forward" --samples "$samples" --threads 4 --require-speedup 1 --require-rss-ratio 1 > "benchmarks/results/coordinate-${name}-n${samples}.json"
        done
    done
    for name in sph_harm vsh_X vsh_Y vsh_Z vsw_M vsw_N vsw_A vsw_rM vsw_rN vsw_rA vcw_M vcw_N vcw_A vcw_rM vcw_rN vcw_rA vpw_M vpw_N vpw_A; do
        for samples in 1 128 4096; do
            uv run --no-sync python scripts/benchmark_cluster.py --workload "wave-${name}-forward" --samples "$samples" --threads 4 --require-speedup 1 --require-rss-ratio 1 > "benchmarks/results/wave-${name}-n${samples}.json"
        done
    done
    for name in vsw_rA vcw_rA vpw_A; do
        for samples in 128 4096; do
            uv run --no-sync python scripts/benchmark_cluster.py --workload "wave-${name}" --samples "$samples" --threads 4 --require-speedup 1 --require-rss-ratio 1 > "benchmarks/results/wave-adjoint-final-${name}-n${samples}.json"
        done
    done
    for name in tl_vsw_A tl_vsw_B tl_vsw_rA tl_vsw_rB tl_vcw tl_vcw_r; do
        for samples in 1 128 4096; do
            uv run --no-sync python scripts/benchmark_cluster.py --workload "polar-${name}-forward" --samples "$samples" --threads 4 --require-speedup 1 --require-rss-ratio 1 > "benchmarks/results/polar-${name}-n${samples}.json"
        done
    done
    for name in tl_vsw_A tl_vsw_rB tl_vcw tl_vcw_r; do
        for samples in 128 4096; do
            uv run --no-sync python scripts/benchmark_cluster.py --workload "polar-${name}" --samples "$samples" --threads 4 --require-speedup 1 --require-rss-ratio 1 > "benchmarks/results/polar-adjoint-${name}-n${samples}.json"
        done
    done
    for name in sw.rotate sw.translate sw.periodic_to_pw sw.periodic_to_cw cw.rotate cw.translate cw.to_sw cw.periodic_to_pw pw.translate pw.to_sw pw.to_cw pw.permute_xyz; do
        for samples in 1 128 4096; do
            uv run --no-sync python scripts/benchmark_cluster.py --workload "namespace-${name}-forward" --samples "$samples" --threads 4 --require-speedup 1 --require-rss-ratio 1 > "benchmarks/results/namespace-${name}-n${samples}.json"
        done
    done
    for name in incgamma intkambe; do
        for samples in 128 4096; do
            uv run --no-sync python scripts/benchmark_cluster.py --workload "$name" --samples "$samples" --threads 4 --require-speedup 1 --require-rss-ratio 1 > "benchmarks/results/${name}-adjoint-n${samples}.json"
        done
    done

# Native geometry and mode metadata helpers, including scalar and broadcast calls.
bench-geometry:
    #!/usr/bin/env bash
    set -euo pipefail
    for name in volume2 volume3 reciprocal2 reciprocal3 refractive_index wave_vec_z; do
        for samples in 1 128 4096; do
            uv run --no-sync python scripts/benchmark_cluster.py --workload "geometry-${name}-forward" --samples "$samples" --threads 4 --require-speedup 1 --require-rss-ratio 1 > "benchmarks/results/geometry-${name}-n${samples}.json"
        done
    done
    for name in cube cubeedge diffr_orders_circle basischange pickmodes firstbrillouin1d firstbrillouin2d firstbrillouin3d; do
        for samples in 1 4 16; do
            uv run --no-sync python scripts/benchmark_cluster.py --workload "geometry-${name}-forward" --samples "$samples" --threads 4 --require-speedup 1 --require-rss-ratio 1 > "benchmarks/results/geometry-${name}-n${samples}.json"
        done
    done

# All scalar Ewald components and direct shells, plus their recorded boundaries.
bench-lattice:
    #!/usr/bin/env bash
    set -euo pipefail
    for prefix in lsum realsum recsum dsum; do
        for family in sw1d sw1d_shift sw2d sw2d_shift sw3d cw1d cw1d_shift cw2d; do
            for samples in 1 128 4096; do
                uv run --no-sync python scripts/benchmark_cluster.py --workload "lattice-${prefix}${family}-forward" --samples "$samples" --threads 4 --require-speedup 1 --require-rss-ratio 1 > "benchmarks/results/lattice-${prefix}${family}-n${samples}.json"
            done
        done
        for family in sw1d_shift sw2d_shift sw3d cw1d_shift cw2d; do
            for samples in 128 4096; do
                uv run --no-sync python scripts/benchmark_cluster.py --workload "lattice-${prefix}${family}" --samples "$samples" --threads 4 --require-speedup 1 --require-rss-ratio 1 > "benchmarks/results/lattice-${prefix}${family}-adjoint-n${samples}.json"
            done
        done
    done

# Python operator workflows and native custom-table and real-degree boundaries.
bench-api:
    #!/usr/bin/env bash
    set -euo pipefail
    for family in sphere cylinder plane; do
        for field in efield hfield dfield bfield gfield ffield; do
            for samples in 1 4096; do
                uv run --no-sync python scripts/benchmark_cluster.py --workload "operator-${family}-${field}-forward" --lmax 3 --samples "$samples" --threads 4 --require-speedup 1 --require-rss-ratio 1 > "benchmarks/results/operator-${family}-${field}-n${samples}.json"
            done
        done
    done
    for pair in '3 1' '3 4' '3 16' '6 4'; do
        read -r order particles <<< "$pair"
        for family in helicity parity; do
            for suffix in '-forward' ''; do
                uv run --no-sync python scripts/benchmark_cluster.py --workload "callback-${family}${suffix}" --lmax "$order" --particles "$particles" --threads 4 --require-speedup 1 --require-rss-ratio 1 > "benchmarks/results/callback-${family}${suffix}-l${order}-p${particles}.json"
            done
        done
    done
    for order in 3 16 64; do
        for samples in 1 128 4096; do
            for suffix in '-forward' ''; do
                uv run --no-sync python scripts/benchmark_cluster.py --workload "angular-fractional${suffix}" --lmax "$order" --samples "$samples" --threads 4 --require-speedup 1 --require-rss-ratio 1 > "benchmarks/results/fractional${suffix}-l${order}-n${samples}.json"
            done
        done
    done

# Public power workflows and the complete recorded native power boundary.
bench-power:
    #!/usr/bin/env bash
    set -euo pipefail
    for samples in 1 8 64 256; do
        for workload in power-tr-forward power-cd-forward power-tr power-translate-forward power-permute-forward; do
            uv run --no-sync python scripts/benchmark_cluster.py --workload "$workload" --samples "$samples" --threads 4 --require-speedup 1 --require-rss-ratio 1 > "benchmarks/results/${workload}-n${samples}.json"
        done
    done

# Run all performance suites sequentially on the same otherwise idle CPU set.
bench-all: bench-performance bench-geometry bench-lattice bench-api bench-power
