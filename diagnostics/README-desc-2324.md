# DESC #2324: fork-only configuration handoff diagnostic

AI-assisted diagnostic preparation, based on PR head
`ff2ef7fee14113e570c5e882c187cd9e918ecbc6`.

The probe compares baseline, standalone `all` normalization, and normalization
plus an explicit JAX configuration handoff. Production files on this branch are
unchanged. Candidate changes are applied only to temporary source copies and
emitted as `candidate.patch` in the CI artifact.

Every case uses a fresh process, real `desc` and `jax` imports, and exact copies
of `desc/__init__.py`, `desc/_version.py`, and `desc/backend.py`. The two modified
modules are checked against their upstream Git blob hashes before patching.
The probe intercepts the first `jnp.linspace` call to observe JAX configuration
BEFORE computation. It does not allocate/query a GPU, initialize the backend,
import the rest of DESC, run an equilibrium solve, or validate CUDA index mapping.

The matrix covers JAX 0.6.2 and 0.9.0.1, both within the inspected requirements.
Each version has 16 scenarios and 3 variants. Green CI means the expected
baseline contract failures were observed and all candidate contracts passed.
It does NOT mean the baseline passes, a full upstream suite passes, or hardware
selection is verified. Unexpected errors fail the workflow.

Covered: `all` with explicit selection, unset visibility, restricted lists,
empty visibility without an explicit index, CPU/TPU noninterference, retaining
programmatic JAX config when no environment selection exists, unchanged CUDA-
wide visibility, and lazy JAX import. The candidate's environment precedence
is a proposal awaiting maintainer review, not an agreed upstream policy.
Already-initialized JAX backends, scheduler/CUDA remapping, invalid device lists,
and physical GPU behavior are deliberately out of scope.

Local validation before upload: syntax checks passed; 16 source-boundary
scenarios on installed JAX 0.9.0.1 gave baseline 8/16, normalization-only 8/16,
and normalization-plus-handoff 16/16. Local source-boundary tests are not these
real-module CI runs: the local environment lacks DESC's import dependencies and
cannot reach the package index.

Only this disposable branch replaces inherited workflows with a read-only,
bounded CPU diagnostic. The default branch, upstream PR, and production sources
are unchanged. No schedule, deployment, secrets, or upstream comment is created.

Sources:
- https://github.com/PlasmaControl/DESC/pull/2324
- https://github.com/PlasmaControl/DESC/blob/ff2ef7fee14113e570c5e882c187cd9e918ecbc6/desc/__init__.py
- https://github.com/PlasmaControl/DESC/blob/ff2ef7fee14113e570c5e882c187cd9e918ecbc6/desc/backend.py
- https://github.com/PlasmaControl/DESC/blob/ff2ef7fee14113e570c5e882c187cd9e918ecbc6/requirements.txt
- https://github.com/PlasmaControl/DESC/blob/ff2ef7fee14113e570c5e882c187cd9e918ecbc6/CONTRIBUTING.rst
