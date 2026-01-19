# Contributing Guidelines

This repository implements the **Noetic Geometry Framework**, originally proposed by **Sar Hamam**, unifying additive/multiplicative transports, Mellin coupling, submersion geometry, and Fisher–Rao pullbacks into a single geometric data oriented library.

## How to Contribute

We welcome contributions in the form of:
- **Bug reports** (please include minimal reproducible examples).
- **Feature proposals** (open an issue before submitting a large PR).
- **Pull requests** (see workflow below).

## Development Workflow

1. **Fork & clone** the repository.
2. **Set up environment**:
   ```bash
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
````

3. **Branching**:

   * `main`: stable releases
   * `dev`: active development
   * feature branches: `feature/<short-name>`

4. **Run tests & checks before commit**:

   ```bash
   pytest --cov=.
   black . && ruff check . && mypy . --ignore-missing-imports
   ```

5. **Commit message style**:

   * `feat:` new feature
   * `fix:` bug fix
   * `test:` testing improvements
   * `docs:` documentation only

## Core Principles

* **Sparse-first numerics**: Always prefer k-NN graphs + CG/Lanczos. No dense eigensolvers for large n.
* **Determinism**: Every function must accept `seed` and respect it.
* **Falsifiability**: New features must include a measurable test (stability, separability, etc.).
* **Reproducibility**: Log seeds, configs, checksums, and artifact hashes.

## Attribution

This framework is based on the theoretical and practical research of **Sar Hamam**.
When extending the codebase, please retain attribution in documentation, commit messages, and papers derived from this work.