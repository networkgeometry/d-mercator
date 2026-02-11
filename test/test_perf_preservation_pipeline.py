import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HAS_VALIDATION_DEPS = all(
    importlib.util.find_spec(pkg) is not None
    for pkg in ("numpy", "pandas", "matplotlib")
)


@unittest.skipUnless(HAS_VALIDATION_DEPS, "Validation dependencies (numpy/pandas/matplotlib) are not installed")
class PerfPreservationPipelineSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.build_dir = cls.repo_root / "build"

    def test_baseline_vs_optimized_smoke(self):
        with tempfile.TemporaryDirectory(prefix="dmercator_perf_preservation_") as tmpdir:
            cmd = [
                sys.executable,
                str(self.repo_root / "python" / "validate_perf_preservation.py"),
                "--build-dir",
                str(self.build_dir),
                "--output-dir",
                tmpdir,
                "--dimension",
                "2",
                "--nodes",
                "64",
                "--seed",
                "12345",
                "--beta",
                "4.5",
                "--rmse-tol",
                "1e-6",
                "--max-abs-tol",
                "1e-5",
                "--skip-build",
            ]
            subprocess.run(cmd, check=True, cwd=self.repo_root)


if __name__ == "__main__":
    unittest.main()
