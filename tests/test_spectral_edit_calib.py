from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from finetune.spectral_edit.calib import _resolve_local_calibration_data_files


class SpectralEditCalibrationTests(unittest.TestCase):
    def test_resolve_local_calibration_snapshot_directory(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_dir = root / "main"
            split_dir.mkdir(parents=True, exist_ok=True)
            target = split_dir / "test-00000-of-00001.parquet"
            target.write_text("placeholder", encoding="utf-8")

            loader_name, files = _resolve_local_calibration_data_files(
                dataset_path=str(root),
                split="test",
                dataset_config="main",
            )

            self.assertEqual(loader_name, "parquet")
            self.assertEqual(files, [str(target)])


if __name__ == "__main__":
    unittest.main()
