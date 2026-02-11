import subprocess
import sys
import tempfile
import unittest
import importlib.util
from pathlib import Path

HAS_VALIDATION_DEPS = all(
    importlib.util.find_spec(pkg) is not None
    for pkg in ("numpy", "pandas", "matplotlib")
)


@unittest.skipUnless(HAS_VALIDATION_DEPS, "Validation dependencies (numpy/pandas/matplotlib) are not installed")
class ValidationPipelineSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.build_dir = cls.repo_root / "build"

    def run_validation(self, dimension: int, nodes: int, seed: int) -> None:
        with tempfile.TemporaryDirectory(prefix=f"dmercator_d{dimension}_") as tmpdir:
            cmd = [
                sys.executable,
                str(self.repo_root / "python" / "validate_sd.py"),
                "--build-dir",
                str(self.build_dir),
                "--output-dir",
                tmpdir,
                "--dimension",
                str(dimension),
                "--nodes",
                str(nodes),
                "--seed",
                str(seed),
                "--pair-samples",
                "2000",
                "--assert-rmse-ref-orig",
                "1e-5",
                "--assert-rmse-ref-truth",
                "2.2",
            ]
            subprocess.run(cmd, check=True, cwd=self.repo_root)

    def test_dimension_1_smoke(self):
        self.run_validation(dimension=1, nodes=180, seed=123)

    def test_dimension_2_smoke(self):
        self.run_validation(dimension=2, nodes=140, seed=456)


if __name__ == "__main__":
    unittest.main()
