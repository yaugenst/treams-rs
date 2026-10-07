# Releasing

`workspace.package.version` in [`Cargo.toml`](https://github.com/yaugenst/treams-rs/blob/217cbe646cd70276ad1622fc61178b512b7faaae/Cargo.toml) sets the
version of both Rust crates and the Python package. Maturin reads it through
the dynamic version in `pyproject.toml`; `Cargo.lock` records the same value.
The Rust crates are internal and are not published to crates.io.

The release build targets CPython 3.12–3.15 on Linux with glibc 2.17 or newer
(x86-64 and arm64), macOS (Intel and Apple silicon), and Windows x86-64,
plus one source distribution. The Python classifiers and `license-files` in
[`pyproject.toml`](https://github.com/yaugenst/treams-rs/blob/217cbe646cd70276ad1622fc61178b512b7faaae/pyproject.toml) define the expected distributions.
Publication requires that every configured wheel builds and passes its
installation checks. The Intel macOS wheels are built on Apple-silicon runners
and checked on Intel runners. PyPy, free-threaded CPython, musllinux, Windows
ARM64, abi3 wheels and conda are outside this release.

## Repository setup

Before the first release, check these settings:

- Make the repository public and enable the protections below.
- Give the Codecov GitHub App access to the repository. The coverage upload
  uses OIDC for the badge and is best-effort. CI enforces coverage independently
  from the saved reports; a Codecov upload failure does not fail `CI Success`.
- Register pending trusted publishers on TestPyPI and PyPI for
  `yaugenst/treams-rs`, workflow `publish-release.yml`, with environments
  `testpypi` and `pypi` respectively. Existing projects use the same publishers.
- Create the `testpypi`, `pypi` and `release` GitHub environments. All three
  accept deployments only from `main`, excluding tags. Require a release
  operator's approval for `release`; a sole operator needs self-review enabled.
- Protect `v*` tags: only release operators may create them, and updates,
  force pushes and deletion are forbidden.
- Set GitHub Pages **Source** to **GitHub Actions**. The documentation workflow
  publishes the complete version tree through a Pages artifact; `gh-pages`
  stores the version history used by mike.
- Enable [private vulnerability reporting](https://github.com/yaugenst/treams-rs/blob/217cbe646cd70276ad1622fc61178b512b7faaae/SECURITY.md).
- Ensure the release operator can send a repository dispatch with
  contents-write access.

## Prepare a version

1. Set the version in `Cargo.toml`, then run `cargo check --workspace` to
   update `Cargo.lock`.
2. Finish the release entry in `CHANGELOG.md`, set its date, and add an empty
   `Unreleased` section. Set the matching `version` and `date-released` in
   `CITATION.cff`.
3. Follow [Development](index.md) to install the locked tools and framework
   dependencies. Run `just ci`, `just check-wheel`, `just licenses-check`,
   `just deny` and `just formal`.
4. Commit the version and release notes, land them on `main`, and wait for
   `CI Success` on that exact push. Check that the version is unused on
   TestPyPI, PyPI and in Git tags.

## Publish

Dispatch from an authorized GitHub CLI session after main CI passes. The
version tag must not exist yet:

```sh
gh api --method POST repos/yaugenst/treams-rs/dispatches \
  -f event_type=release \
  -f 'client_payload[tag]=vX.Y.Z'
```

The [publication workflow](https://github.com/yaugenst/treams-rs/blob/217cbe646cd70276ad1622fc61178b512b7faaae/.github/workflows/publish-release.yml)
fixes the candidate to the current `main` commit and requires a successful
`CI Success` from a push to that commit. It builds and checks the complete
set of wheels and source distribution, checks metadata and local-path removal,
and smoke-tests every distribution in clean environments. It then uploads the
candidate to TestPyPI and tests a clean installation from there. That step is
advisory: TestPyPI has no service-level agreement, so a failed upload or
installation there is reported in the workflow summary but does not block
promotion. To skip it, add `-f 'client_payload[skip_testpypi]=true'` to the
dispatch.

When `Approve production release` starts waiting, use the candidate revision
in the workflow summary to create and push the annotated tag:

```sh
git tag -a vX.Y.Z CANDIDATE_SHA -m "treams-rs X.Y.Z"
git push origin vX.Y.Z
```

Then approve the `release` environment. The workflow checks that the tag
still points to the tested candidate before publishing versioned docs, the
GitHub Release and the same distributions on PyPI. It checks a clean PyPI
installation afterward. Do not create the tag before the approval job waits:
the missing tag also prevents publication if environment protection is lost.

Avoid merging to `main` until publication finishes. Documentation deployments
share a queue; GitHub may cancel an older waiting job when a newer job queues.
If the documentation job is cancelled, rerun failed jobs; the tag check runs
again.

After the workflow finishes, confirm:

- the GitHub Release contains every wheel, one source distribution, checksums
  and the provenance manifest;
- the clean PyPI installation check passed;
- `/X.Y.Z/` and `/latest/` serve the released documentation, including rustdoc,
  and the site root opens the latest release.

Never replace a published file or move a version tag. A candidate whose files
reached TestPyPI consumes its version there too, since filenames cannot be
reused: if it turns out faulty, fix the issue and choose the next version. A
TestPyPI failure before any upload, such as an outage, consumes nothing.

## Documentation site

The reusable [Docs workflow](https://github.com/yaugenst/treams-rs/blob/217cbe646cd70276ad1622fc61178b512b7faaae/.github/workflows/docs.yml) checks pull
requests and publishes `dev` from `main`. A release publishes `X.Y.Z` and
updates `latest`; the site root follows `latest` after the first release.
Every version includes its Rust reference and Markdown sources. Source links
refer to the documented commit.

To correct released documentation without changing the package, run
[Deploy docs](https://github.com/yaugenst/treams-rs/blob/217cbe646cd70276ad1622fc61178b512b7faaae/.github/workflows/deploy-docs.yml) from `main`. Set `version` to an
existing release `X.Y.Z` and `source_revision` to the full commit SHA of the
correction. The workflow requires the existing release and immutable tag.
It replaces that version's documentation and updates `latest` only for the
newest release. It does not change the package files or release tag.

## Check a wheel locally

`just check-wheel` builds an optimized wheel, checks it for local build paths,
and runs [`smoke_wheel_install.py`](https://github.com/yaugenst/treams-rs/blob/217cbe646cd70276ad1622fc61178b512b7faaae/scripts/smoke_wheel_install.py) in
clean environments with NumPy only, Advect, HIPS Autograd and HDF5. CI runs the same smoke
checks for every configured wheel and for a wheel rebuilt from the source
distribution. Local success does not replace those platform checks.
