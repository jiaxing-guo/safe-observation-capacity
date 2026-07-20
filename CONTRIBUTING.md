# Contributing

Use Python 3.11 or newer, a current stable Rust toolchain, and `uv`.

```bash
make install
make check
```

Keep Python under `src/safe_observation`, Rust crates under `crates`, reusable experiment entry points under `scripts`, and small checked-in configurations under `configs`.

Run `make format` before committing. New behavior should include focused tests. Source code and checked-in configurations must remain free of comments so the public release stays consistent with its publication policy.

Do not commit generated results, experiment artifacts, paper sources, logs, cluster scheduler definitions, or machine-specific paths. The repository ignore rules cover the standard output locations; use a local directory outside the repository for anything else.
