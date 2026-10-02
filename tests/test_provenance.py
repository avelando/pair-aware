import subprocess
import unittest
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import torch

import src.experiments.provenance as provenance_module


class ProvenanceTest(unittest.TestCase):
    def test_utc_now_returns_timezone_aware_utc_timestamp(self):
        timestamp = provenance_module.utc_now()
        parsed = datetime.fromisoformat(
            timestamp
        )

        self.assertIsNotNone(
            parsed.tzinfo
        )

        self.assertEqual(
            parsed.utcoffset(),
            timezone.utc.utcoffset(
                parsed
            ),
        )

    def test_missing_package_version_returns_none(self):
        with patch.object(
            provenance_module,
            "version",
            side_effect=PackageNotFoundError(
                "missing-package"
            ),
        ):
            result = provenance_module._package_version(
                "missing-package"
            )

        self.assertIsNone(
            result
        )

    def test_git_provenance_records_commit_branch_and_dirty_state(self):
        with patch.object(
            provenance_module,
            "_run_git",
            side_effect=[
                "abc123",
                "refactor/python-scripts",
                " M src/config.py",
            ],
        ):
            result = provenance_module.get_git_provenance()

        self.assertEqual(
            result,
            {
                "git_available": True,
                "git_commit": "abc123",
                "git_branch": "refactor/python-scripts",
                "git_dirty": True,
            },
        )

    def test_git_provenance_reports_unavailable_git(self):
        error = subprocess.CalledProcessError(
            1,
            ["git"],
        )

        with patch.object(
            provenance_module,
            "_run_git",
            side_effect=error,
        ):
            result = provenance_module.get_git_provenance()

        self.assertEqual(
            result,
            {
                "git_available": False,
                "git_commit": None,
                "git_branch": None,
                "git_dirty": None,
            },
        )

    def test_environment_metadata_records_runtime_versions(self):
        package_versions = {
            "accelerate": "1.14.0",
            "numpy": "2.5.3",
            "pandas": "2.3.2",
            "scikit-learn": "1.9.0",
        }

        with patch.object(
            provenance_module.platform,
            "python_version",
            return_value="3.12.0",
        ), patch.object(
            provenance_module.torch.cuda,
            "get_device_name",
            return_value="NVIDIA GeForce RTX 5090",
        ), patch.object(
            provenance_module,
            "_package_version",
            side_effect=lambda name: package_versions[
                name
            ],
        ):
            metadata = provenance_module.get_environment_metadata(
                torch.device("cuda")
            )

        self.assertEqual(
            metadata["python_version"],
            "3.12.0",
        )

        self.assertEqual(
            metadata["device"],
            "cuda",
        )

        self.assertEqual(
            metadata["gpu"],
            "NVIDIA GeForce RTX 5090",
        )

        self.assertEqual(
            metadata["accelerate_version"],
            "1.14.0",
        )

        self.assertEqual(
            metadata["numpy_version"],
            "2.5.3",
        )

        self.assertEqual(
            metadata["pandas_version"],
            "2.3.2",
        )

        self.assertEqual(
            metadata["scikit_learn_version"],
            "1.9.0",
        )

        self.assertIn(
            "cuda_version",
            metadata,
        )

        self.assertIn(
            "torch_version",
            metadata,
        )

        self.assertIn(
            "transformers_version",
            metadata,
        )


if __name__ == "__main__":
    unittest.main()