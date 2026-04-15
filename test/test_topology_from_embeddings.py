import importlib.util
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HAS_TOPOLOGY_DEPS = all(
    importlib.util.find_spec(pkg) is not None
    for pkg in ("numpy", "matplotlib", "scipy")
)


@unittest.skipUnless(HAS_TOPOLOGY_DEPS, "Topology validation dependencies are not installed")
class TopologyFromEmbeddingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(cls.repo_root))
        from python import topology_from_embeddings as topology_module

        cls.topology = topology_module
        cls.script_path = cls.repo_root / "python" / "topology_from_embeddings.py"

    def create_stub_generator(self, directory: Path) -> Path:
        generator_path = directory / "stub_generate_sd.py"
        generator_path.write_text(
            """#!/usr/bin/env python3
import sys
from pathlib import Path


def main():
    args = sys.argv[1:]
    out_root = None
    hidden_vars = None
    index = 0
    while index < len(args):
        token = args[index]
        if token == "-o":
            out_root = Path(args[index + 1])
            index += 2
            continue
        if token in {"-d", "-b", "-m", "-s"}:
            index += 2
            continue
        if token in {"-n", "-t"}:
            index += 1
            continue
        hidden_vars = Path(token)
        index += 1

    if out_root is None or hidden_vars is None:
        raise SystemExit("missing required arguments")

    names = []
    with hidden_vars.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            names.append(stripped.split()[0])

    out_root.parent.mkdir(parents=True, exist_ok=True)
    edge_path = out_root.with_suffix(".edge")
    with edge_path.open("w", encoding="utf-8") as handle:
        handle.write("# source target\\n")
        for index, source in enumerate(names):
            target = names[(index + 1) % len(names)]
            handle.write(f"{source} {target}\\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
            encoding="utf-8",
        )
        generator_path.chmod(generator_path.stat().st_mode | stat.S_IEXEC)
        return generator_path

    def write_inf_coord(self, path: Path, node_count: int) -> None:
        lines = [
            "#   - beta:              2",
            "#   - mu:                0.5",
        ]
        for index in range(node_count):
            name = f"v{index}"
            kappa = 1.0 + index
            theta = (2.0 * math.pi * index) / node_count
            radial = 2.0 + index
            lines.append(f"{name} {kappa:.6f} {theta:.6f} {radial:.6f}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_cycle_edge_list(self, path: Path, node_count: int) -> None:
        with path.open("w", encoding="utf-8") as handle:
            handle.write("# source target\n")
            for index in range(node_count):
                source = f"v{index}"
                target = f"v{(index + 1) % node_count}"
                handle.write(f"{source} {target}\n")

    def create_case(
        self,
        artifacts_dir: Path,
        case_name: str,
        node_count: int,
        embeddings,
    ) -> None:
        case_dir = artifacts_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        self.write_cycle_edge_list(case_dir / "synthetic_sd_GC.edge", node_count)

        for backend, sample_count in embeddings:
            root = case_dir / f"embed_{backend}_{sample_count}"
            self.write_inf_coord(root.with_suffix(".inf_coord"), node_count)
            timing = {
                "mode": "optimized",
                "backend": backend,
                "dimension": 1,
                "seed": int(case_name.split("seed", 1)[1]),
                "refine_negative_samples": 0 if sample_count == "exact" else int(sample_count),
                "nb_vertices": node_count,
                "nb_edges": node_count,
                "total_time_ms": 10.0,
                "refining_positions_ms": 2.0,
                "wall_time_ms": 11.0,
            }
            root.with_suffix(".timing.json").write_text(json.dumps(timing), encoding="utf-8")
            metrics = {
                "objective": -1.0,
                "c_score": 1.0 if backend == "cpu" else 0.99,
                "aligned_coord_corr": None,
            }
            root.with_suffix(".metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    def create_job_layout(self, root: Path, nested: bool) -> Path:
        job_dir = root / ("nested_job" if nested else "direct_job")
        artifacts_dir = job_dir / "benchmark" / "artifacts" if nested else job_dir / "artifacts"
        self.create_case(
            artifacts_dir,
            case_name="d1_n4_seed1004",
            node_count=4,
            embeddings=[("cpu", "exact"), ("gpu", "16")],
        )
        self.create_case(
            artifacts_dir,
            case_name="d1_n5_seed1005",
            node_count=5,
            embeddings=[("cpu", "exact"), ("gpu", "16")],
        )
        return job_dir

    def test_parse_helpers_and_seed(self):
        case_info = self.topology.parse_case_directory_name("d1_n5000_seed1005017")
        self.assertEqual(case_info["dimension"], 1)
        self.assertEqual(case_info["size"], 5000)
        self.assertEqual(case_info["graph_seed"], 1005017)

        embed_info = self.topology.parse_embedding_root(Path("embed_gpu_256.inf_coord"))
        self.assertEqual(embed_info["backend"], "gpu")
        self.assertEqual(embed_info["sample_count"], "256")

        record = {
            "graph_seed": 1005017,
            "backend": "gpu",
            "sample_count": "256",
        }
        self.assertEqual(self.topology.synthetic_seed(record), 1005017 + 1000 + 256)

    def test_discover_job_records_for_direct_and_nested_layouts(self):
        with tempfile.TemporaryDirectory(prefix="topology_discovery_") as tmpdir:
            temp_root = Path(tmpdir)
            direct_job = self.create_job_layout(temp_root, nested=False)
            nested_job = self.create_job_layout(temp_root, nested=True)

            direct_records = self.topology.discover_job_records(direct_job)
            nested_records = self.topology.discover_job_records(nested_job)

            self.assertEqual(len(direct_records), 4)
            self.assertEqual(len(nested_records), 4)

            direct_pairs = {(record["backend"], record["sample_count"]) for record in direct_records}
            nested_pairs = {(record["backend"], record["sample_count"]) for record in nested_records}
            self.assertEqual(direct_pairs, {("cpu", "exact"), ("gpu", "16")})
            self.assertEqual(nested_pairs, {("cpu", "exact"), ("gpu", "16")})

            for record in direct_records + nested_records:
                self.assertTrue(record["edge_file"].endswith("synthetic_sd_GC.edge"))
                self.assertTrue(record["inf_coord_file"].endswith(".inf_coord"))

    def test_job_dir_smoke_generates_reports_caches_and_plots(self):
        with tempfile.TemporaryDirectory(prefix="topology_smoke_") as tmpdir:
            temp_root = Path(tmpdir)
            job_dir = self.create_job_layout(temp_root, nested=False)
            generator_path = self.create_stub_generator(temp_root)

            subprocess.run(
                [
                    sys.executable,
                    str(self.script_path),
                    "--job-dir",
                    str(job_dir),
                    "--generator-binary",
                    str(generator_path),
                    "--min-degree-count",
                    "1",
                ],
                check=True,
                cwd=self.repo_root,
            )

            output_dir = job_dir / "topology_validation"
            self.assertTrue((output_dir / "topology_from_embeddings.json").exists())
            self.assertTrue((output_dir / "topology_from_embeddings.csv").exists())
            self.assertTrue((output_dir / "topology_from_embeddings_report.md").exists())
            self.assertTrue((output_dir / "plots" / "topology_metrics_vs_n.png").exists())
            self.assertTrue((output_dir / "plots" / "degree_ccdf_vs_original.png").exists())
            self.assertTrue((output_dir / "plots" / "clustering_spectrum_ck_vs_original.png").exists())
            self.assertTrue((output_dir / "plots" / "adjacency_spectrum_vs_original.png").exists())

            summary = json.loads((output_dir / "topology_from_embeddings.json").read_text(encoding="utf-8"))
            self.assertEqual(len(summary), 4)
            for record in summary:
                self.assertEqual(record["degree_ks"], 0.0)
                self.assertEqual(record["avg_clustering_abs_diff"], 0.0)
                self.assertEqual(record["transitivity_abs_diff"], 0.0)
                self.assertEqual(record["ck_weighted_rmse"], 0.0)
                self.assertEqual(record["spectrum_rel_rmse"], 0.0)
                self.assertEqual(record["edge_count_abs_diff"], 0)
                self.assertTrue(Path(record["synthetic_edge_file"]).exists())

            original_cache = output_dir / "properties" / "d1_n4_seed1004" / "original.npz"
            synthetic_cache = output_dir / "properties" / "d1_n4_seed1004" / "cpu_exact.npz"
            self.assertTrue(original_cache.exists())
            self.assertTrue(synthetic_cache.exists())


if __name__ == "__main__":
    unittest.main()
