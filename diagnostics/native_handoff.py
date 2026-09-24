"""Fork-only DESC #2324 boundary probe using real DESC and JAX imports.

AI-assisted. Stops at the first jnp.linspace call: no GPU query, kernel, or full
DESC computation. A successful workflow means expected baseline failures and
passing candidate contracts, NOT a passing upstream test suite.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

BASE = "ff2ef7fee14113e570c5e882c187cd9e918ecbc6"
BLOBS = {"__init__.py": "cebfe480cc5f4c7fce3e5abe1f871c715e36a0e0",
         "backend.py": "700c4df4dbf733b89d558ea19a5bf880d16493db"}
VARIANTS = ("baseline", "normalization_only", "normalization_and_handoff")
CASES = [
    ("all_select_0", "gpu", "all", 0, None, "0"),
    ("all_select_1", "gpu", "all", 1, None, "1"),
    ("unset_select_0", "gpu", None, 0, None, "0"),
    ("unset_select_1", "gpu", None, 1, None, "1"),
    ("restricted_first", "gpu", "5,2", 0, None, "5"),
    ("restricted_second", "gpu", "5,2", 1, None, "2"),
    ("restricted_no_selection", "gpu", "5,2", None, None, "5,2"),
    ("all_no_selection", "gpu", "all", None, None, "all"),
    ("empty_no_selection", "gpu", "", None, None, ""),
    ("unset_no_selection", "gpu", None, None, None, "all"),
    ("preserve_programmatic_without_env", "gpu", None, None, "2", "2"),
    ("cpu_preserves_programmatic", "cpu", None, None, "2", "2"),
    ("cpu_ignores_gpu_env", "cpu", "5,2", None, "2", "2"),
    ("cpu_ignores_all_env", "cpu", "all", None, "2", "2"),
    ("tpu_ignores_gpu_env", "tpu", "5,2", None, "2", "2"),
    ("cpu_without_programmatic", "cpu", "5,2", None, None, "all"),
]
EXPECTED_FAILURES = {CASES[i][0] for i in [0, 1, 2, 3, 4, 5, 6, 8]}
BRIDGE = '''            if (
                desc_config["kind"] == "gpu"
                and "JAX_CUDA_VISIBLE_DEVICES" in os.environ
            ):
                jax_config.update(
                    "jax_cuda_visible_devices", os.environ["JAX_CUDA_VISIBLE_DEVICES"]
                )
'''


def patched(text, filename, variant):
    if filename == "__init__.py" and variant != "baseline":
        old = "            visible = [i for i in visible if i]\n"
        assert text.count(old) == 1
        text = text.replace(old, old + '            visible = [] if visible == ["all"] else visible\n')
    if filename == "backend.py" and variant == "normalization_and_handoff":
        old = '            jax_config.update("jax_enable_x64", True)\n'
        assert text.count(old) == 1
        text = text.replace(old, BRIDGE + old)
    return text


def native_case(source, variant, number):
    name, kind, visible, gpuid, preexisting, expected = CASES[number]
    for key in tuple(os.environ):
        if key.startswith(("JAX_", "CUDA_", "NVIDIA_", "DESC_", "XLA_")):
            del os.environ[key]
    os.environ["CUDA_VISIBLE_DEVICES"] = "7,9"
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "GPU-probe-placeholder"
    if visible is not None:
        os.environ["JAX_CUDA_VISIBLE_DEVICES"] = visible
    result = {"case": name, "variant": variant, "expected": expected,
              "reached_first_operation": False, "passed": False, "stage": "setup"}
    with tempfile.TemporaryDirectory(prefix="desc-handoff-") as directory:
        package = Path(directory) / "desc"
        package.mkdir()
        # Unmodified dependency and exact production modules, not replacement stubs.
        for filename in ("__init__.py", "_version.py", "backend.py"):
            raw = (source / "desc" / filename).read_bytes()
            if filename in BLOBS:
                digest = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
                assert digest == BLOBS[filename], (filename, digest)
            (package / filename).write_text(patched(raw.decode(), filename, variant))
        sys.path.insert(0, directory)
        try:
            import desc
            assert Path(desc.__file__).resolve().parent == package.resolve()
            result["stage"] = "set_device"
            desc.set_device(kind, gpuid=gpuid)
            result["lazy_import_preserved"] = "jax" not in sys.modules
            result["environment_visibility"] = os.environ.get("JAX_CUDA_VISIBLE_DEVICES")
            result["stage"] = "jax_import"
            import jax
            import jax.numpy as jnp
            result["jax_version"] = jax.__version__
            if preexisting is not None:
                jax.config.update("jax_cuda_visible_devices", preexisting)

            class FirstOperation(BaseException):
                pass

            def stop_before_compute(*args, **kwargs):
                result["reached_first_operation"] = True
                result["actual"] = jax.config.read("jax_cuda_visible_devices")
                result["cuda_unchanged"] = os.environ["CUDA_VISIBLE_DEVICES"] == "7,9"
                result["nvidia_unchanged"] = os.environ["NVIDIA_VISIBLE_DEVICES"] == "GPU-probe-placeholder"
                raise FirstOperation

            original = jnp.linspace
            jnp.linspace = stop_before_compute
            result["stage"] = "backend_import"
            try:
                import desc.backend
            except FirstOperation:
                result["stage"] = "first_operation_boundary"
            finally:
                jnp.linspace = original
            result["passed"] = (
                result["reached_first_operation"] and result["actual"] == expected
                and result["lazy_import_preserved"] and result["cuda_unchanged"]
                and result["nvidia_unchanged"]
            )
        except Exception as exc:
            result.update(error_type=type(exc).__name__, error=str(exc))
    print(json.dumps(result, sort_keys=True))


def main():
    if len(sys.argv) == 5 and sys.argv[1] == "--child":
        native_case(Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
        return 0
    source = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    out = source / "diagnostic-results"
    out.mkdir(exist_ok=True)
    diff = []
    for filename in BLOBS:
        text = (source / "desc" / filename).read_text()
        diff.extend(difflib.unified_diff(text.splitlines(True),
            patched(text, filename, VARIANTS[-1]).splitlines(True),
            fromfile=f"a/desc/{filename}", tofile=f"b/desc/{filename}"))
    (out / "candidate.patch").write_text("".join(diff))
    def run(task):
        variant, number = task
        p = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", str(source), variant, str(number)],
                           capture_output=True, text=True, timeout=45)
        if p.returncode:
            raise RuntimeError(f"Child infrastructure failure: {p.returncode}\n{p.stderr}")
        return json.loads(p.stdout.strip().splitlines()[-1])
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, [(v, i) for v in VARIANTS for i in range(len(CASES))]))
    (out / "native-results.json").write_text(json.dumps({"base": BASE,
        "scope": "Real DESC and JAX imports, stop before first backend operation; no hardware or full-suite test",
        "results": results}, indent=2) + "\n")
    valid = True
    for variant in VARIANTS:
        group = [r for r in results if r["variant"] == variant]
        failures = {r["case"] for r in group if not r["passed"]}
        expected = set() if variant == VARIANTS[-1] else EXPECTED_FAILURES
        print(f"{variant}: {len(group) - len(failures)}/{len(group)} contracts satisfied")
        valid &= failures == expected
        for r in group:
            print(json.dumps(r, sort_keys=True))
            expected_index_error = variant == "baseline" and r["case"] == "all_select_1"
            if expected_index_error:
                valid &= r.get("error_type") == "IndexError" and r["stage"] == "set_device"
            else:
                valid &= r["reached_first_operation"] and "error_type" not in r
    print("Expected baseline failures plus passing candidate:", bool(valid))
    return 0 if valid else 1

if __name__ == "__main__":
    raise SystemExit(main())
