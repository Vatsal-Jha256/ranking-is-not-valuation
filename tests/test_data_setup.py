"""Data setup checks use mocked HTTP responses and small local input files."""

from __future__ import annotations
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


setup = module("data_setup_under_test", ROOT / "scripts/prepare_data.py")
driver = module("new_data_under_test", ROOT / "scripts/run_new_data.py")


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / "ec3"
        self.args = SimpleNamespace(
            output_dir=self.out, resume=False, page_size=2, max_pages=3, delay=0
        )

    def run_fetch(self, pages):
        with (
            patch.object(setup, "api_session"),
            patch.object(setup, "get_page", side_effect=pages),
            redirect_stdout(io.StringIO()),
        ):
            setup.fetch_ec3(self.args)

    def test_complete_download_writes_final_data_and_state(self):
        self.run_fetch([([{"id": "a"}, {"id": "b"}], "2"), ([], "2")])
        self.assertEqual(len((self.out / "readymix.jsonl").read_text().splitlines()), 2)
        self.assertTrue(
            json.loads((self.out / "readymix.download.json").read_text())["complete"]
        )
        self.assertEqual(
            json.loads((self.out / "readymix.state.json").read_text()),
            {"next_page": 2, "n": 2},
        )

    def test_page_limit_never_labels_partial_download_complete(self):
        self.args.max_pages = 1
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            self.run_fetch([([{"id": "a"}], None)])
        self.assertFalse((self.out / "readymix.jsonl").exists())
        self.assertFalse(
            json.loads((self.out / "readymix.download.json").read_text())["complete"]
        )

    def test_interrupted_download_resumes_without_duplicating_completed_page(self):
        with self.assertRaises(RuntimeError):
            self.run_fetch([([{"id": "a"}], None), RuntimeError("network unavailable")])
        self.args.resume = True
        self.run_fetch([([{"id": "b"}], None), ([], None)])
        values = [
            json.loads(x)["id"]
            for x in (self.out / "readymix.jsonl").read_text().splitlines()
        ]
        self.assertEqual(values, ["a", "b"])

    def test_empty_page_before_reported_total_fails(self):
        with self.assertRaisesRegex(RuntimeError, "before reported total"):
            self.run_fetch([([{"id": "a"}], "4"), ([], "4")])
        self.assertFalse((self.out / "readymix.jsonl").exists())

    def test_existing_snapshot_is_not_overwritten(self):
        self.out.mkdir()
        target = self.out / "readymix.jsonl"
        target.write_text("original")
        with self.assertRaises(FileExistsError):
            self.run_fetch([])
        self.assertEqual(target.read_text(), "original")

    def test_authentication_error_is_not_retried(self):
        session = MagicMock()
        session.get.return_value.status_code = 401
        with self.assertRaisesRegex(PermissionError, "expiry"):
            setup.get_page(session, 1, 1)
        self.assertEqual(session.get.call_count, 1)

    def test_rate_limit_is_retried(self):
        limited = SimpleNamespace(status_code=429, headers={"Retry-After": "0"})
        ok = MagicMock(status_code=200, headers={})
        ok.json.return_value = [{"id": "a"}]
        session = MagicMock()
        session.get.side_effect = [limited, ok]
        with patch.object(setup.time, "sleep"):
            batch, _ = setup.get_page(session, 1, 1)
        self.assertEqual(batch, [{"id": "a"}])
        self.assertEqual(session.get.call_count, 2)


class PreprocessingTests(unittest.TestCase):
    def test_ec3_filtering_uses_shared_code_and_keeps_missing_features(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            valid = dict(
                concrete_compressive_strength_28d="30 MPa",
                declared_unit="1 m3",
                mass_per_declared_unit="2400 kg",
                gwp="300 kgCO2e",
                density="2400 kg/m3",
                description="",
                name="mix",
                mfr_name="supplier",
                plant_name="plant",
                created_on="2025-01-01",
            )
            rows = [
                valid,
                valid.copy(),
                dict(valid, name="bad", gwp="0 kgCO2e"),
                dict(valid, name="missing", concrete_compressive_strength_28d=None),
            ]
            source = root / "data.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            out = root / "processed"
            with redirect_stdout(io.StringIO()):
                setup.preprocess_ec3(SimpleNamespace(input=source, output_dir=out))
            report = json.loads((out / "readymix_preprocessing.json").read_text())
            self.assertEqual(report["retained_rows"], 1)
            self.assertEqual(sum(report["drops"].values()), 3)
            self.assertFalse(report["global_imputation_applied"])
            self.assertFalse(report["matches_frozen_snapshot"])
            features = pd.read_csv(out / "readymix_features.csv")
            self.assertTrue(features["lat"].isna().all())
            self.assertNotIn("gwp", features.columns)

    def test_clf_rebuild_ignores_existing_repository_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.xlsx"
            pd.DataFrame(
                [
                    dict(
                        project_index=1,
                        life_cycle_stage="A1-A3",
                        mat_group="Concrete",
                        mat_type="4000 psi",
                        inv_mass=1000,
                        gwp=100,
                    ),
                    dict(
                        project_index=2,
                        life_cycle_stage="A4",
                        mat_group="Concrete",
                        mat_type="4000 psi",
                        inv_mass=1000,
                        gwp=100,
                    ),
                ]
            ).to_excel(source, index=False)
            output = root / "cache.csv"
            with redirect_stdout(io.StringIO()):
                setup.prepare_clf(SimpleNamespace(source=source, output=output))
            frame = pd.read_csv(output)
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.project_index.tolist(), [1])
            self.assertAlmostEqual(frame.strength_mpa.iloc[0], 4000 / 145.0377)


class NewDataRunTests(unittest.TestCase):
    def test_new_run_preserves_release_results_and_records_input_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            root.mkdir()
            subprocess.run(["git", "init", "-q", str(root)], check=True)

            def write(name, text):
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text)

            write("results/released.json", '{"original": true}\n')
            write(
                "experiments/demo.py",
                "import json,os\nfrom pathlib import Path\nassert 'EC3_API_KEY' not in os.environ\nPath('results/new.json').write_text(json.dumps({'new':True,'evaluations':[{}],'metadata':{'complete':True}}))\n",
            )
            write("figures/gen_submission_figure.py", "pass\n")
            write("figures/gen_appendix_figures.py", "pass\n")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.invalid",
                    "commit",
                    "-qm",
                    "Fixture",
                ],
                check=True,
            )
            for name in driver.INPUTS:
                write(name, "input bytes")
            output = Path(temp) / "new-run"
            with (
                patch.object(driver, "ROOT", root),
                patch.object(driver, "RUNNERS", ("experiments/demo.py",)),
                patch.object(
                    driver, "EXPECTED_RESULTS", {"experiments/demo.py": ("new", 1)}
                ),
                patch.object(
                    sys, "argv", ["run_new_data.py", "--output-dir", str(output)]
                ),
                patch.dict("os.environ", {"EC3_API_KEY": "test-only"}),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(driver.main(), 0)
            self.assertTrue((root / "results/released.json").exists())
            self.assertTrue((output / "reference-results/released.json").exists())
            self.assertTrue((output / "results/new.json").exists())
            self.assertFalse((output / ".env").exists())
            report = json.loads((output / "new-data-run.json").read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(set(report["inputs"]), set(driver.INPUTS))
            self.assertIn("not a frozen-result reproduction", report["kind"])


if __name__ == "__main__":
    unittest.main()
