set shell := ["bash", "-euo", "pipefail", "-c"]

build-ext:
    uv run --no-sync maturin develop -m crates/treams-py/Cargo.toml

build-ext-release:
    uv run --no-sync maturin develop --release -m crates/treams-py/Cargo.toml

rust-fmt-check:
    cargo fmt --check --all

rust-lint:
    PYO3_PYTHON="$PWD/.venv/bin/python" cargo clippy --workspace --all-targets --all-features -- -D warnings

rust-test:
    PYO3_PYTHON="$PWD/.venv/bin/python" cargo test -p treams-core

py-format-check:
    uv run ruff format --check .

py-lint:
    uv run ruff check .

py-types:
    uv run pyrefly check --summarize-errors

dependency-lock-check:
    uv lock --check
    cargo metadata --locked --format-version 1 --no-deps > /dev/null

file-hygiene:
    uv run pre-commit run --all-files --hook-stage manual

check: file-hygiene dependency-lock-check rust-fmt-check rust-lint py-format-check py-lint py-types

test-py: build-ext
    uv run --no-sync pytest

test: rust-test test-py

verify: check test

ci: verify
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps

build-wheel:
    uv run --no-sync maturin build --release --locked --out dist

check-wheel: build-wheel
    #!/usr/bin/env bash
    set -euo pipefail
    env_dir=$(mktemp -d)
    trap 'rm -rf "$env_dir"' EXIT
    uv venv --python .venv/bin/python "$env_dir"
    wheel=$(ls -t dist/*.whl | head -n 1)
    uv pip install --python "$env_dir/bin/python" "$wheel" advect
    "$env_dir/bin/python" scripts/check_wheel.py
    uv pip install --python "$env_dir/bin/python" "${wheel}[io]"
    "$env_dir/bin/python" - <<'PY'
    import h5py
    import numpy as np
    import treams_rs as tr
    from treams_rs import io
    sphere = tr.TMatrix.sphere(1, 1.3, 0.2, [3, (1.3, 1.1, 0.08)])
    cluster = tr.TMatrix.cluster([sphere, sphere], [[0, 0, 0], [0.7, 0.2, 0.1]])
    with h5py.File("memory.h5", "w", driver="core", backing_store=False) as handle:
        io.save_hdf5(handle, cluster)
        loaded = io.load_hdf5(handle, lunit="um")
        np.testing.assert_allclose(loaded.array, cluster.array)
        np.testing.assert_allclose(loaded.basis.positions, cluster.basis.positions * 1e-3)
        np.testing.assert_allclose(loaded.k0, cluster.k0 * 1e3)
        assert loaded.material == cluster.material
    print("Clean wheel: optional HDF5 chirality, origins and units round trip passed")
    PY

# Run on an idle performance host; hosted CI machines have uncontrolled CPU sharing.
bench-performance: build-ext-release
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p benchmarks/results
    for order in 128 512; do
        for columns in 1 8; do
            uv run --no-sync python scripts/benchmark_cluster.py --workload internal-field-forward --particles 1 --lmax "$order" --samples "$columns" --threads 4 --require-speedup 1 --require-rss-ratio 1 > "benchmarks/results/internal-forward-l${order}-p${columns}.json"
            uv run --no-sync python scripts/benchmark_cluster.py --workload internal-field --particles 1 --lmax "$order" --samples "$columns" --threads 4 --require-speedup 1 > "benchmarks/results/internal-adjoint-l${order}-p${columns}.json"
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
