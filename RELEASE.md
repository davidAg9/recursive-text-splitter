# Release Process

This guide explains how to cut a new release of `recursive-text-splitter` and publish it to [crates.io](https://crates.io).

## Prerequisites

1. **GitHub CLI** (`gh`) — [install](https://cli.github.com/) and authenticate:
   ```bash
   gh auth login
   ```

2. **crates.io API token** — stored as GitHub secret `CARGO_REGISTRY_TOKEN`:
   - Get it from https://crates.io/me
   - Add it to repo settings → Secrets and variables → Actions → New repository secret

3. **Cargo installed** (obviously).

## Quick Release (Tag + CI)

The GitHub Actions workflow automatically publishes to crates.io on every tag push matching `v*`:

```bash
# 1. Bump version in Cargo.toml + Cargo.lock
cargo set-version 0.1.1

# 2. Commit
git add Cargo.toml Cargo.lock
git commit -m "Release v0.1.1"

# 3. Tag and push (CI will publish to crates.io)
git tag v0.1.1
git push origin main --tags
```

The [release workflow](.github/workflows/release.yml) will:
- Build with all features
- Run tests (default + rayon)
- Run clippy with `-D warnings`
- Publish to crates.io

## Manual Release

```bash
cargo publish
```

## Before You Release — Checklist

- [ ] `cargo fmt --check`
- [ ] `cargo test` (default features)
- [ ] `cargo test --features rayon` (parallel tests)
- [ ] `cargo clippy --all-features -- -D warnings`
- [ ] `cargo publish --dry-run`
- [ ] README builds correctly on docs.rs
- [ ] LICENSE file is MIT (matches upstream Python project)
