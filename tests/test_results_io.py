import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.config import (
    EXPECTED_PAIR_COUNTS,
    EXPECTED_SPLIT_COUNTS,
)
from src.results.io import (
    is_run_completed,
    mark_run_completed,
    prepare_run_directory,
    write_dataframe,
    write_history,
    write_json,
)


class ResultsIOTest(unittest.TestCase):
    def test_completed_run_requires_all_valid_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            paths = prepare_run_directory(
                run_dir
            )

            write_json(
                {
                    "status": "completed",
                },
                paths["metadata"],
            )

            write_json(
                {
                    "test_instance": {},
                    "test_pair": {},
                },
                paths["metrics"],
            )

            write_history(
                [
                    {
                        "epoch": 1,
                    }
                ],
                paths["history"],
            )

            write_dataframe(
                pd.DataFrame(
                    {
                        "id": range(
                            EXPECTED_SPLIT_COUNTS[
                                "test"
                            ]
                        )
                    }
                ),
                paths["predictions"],
            )

            write_dataframe(
                pd.DataFrame(
                    {
                        "pair_id": range(
                            EXPECTED_PAIR_COUNTS[
                                "test"
                            ]
                        )
                    }
                ),
                paths[
                    "pair_predictions"
                ],
            )

            mark_run_completed(
                run_dir
            )

            self.assertTrue(
                is_run_completed(
                    run_dir
                )
            )

            paths["metrics"].unlink()

            self.assertFalse(
                is_run_completed(
                    run_dir
                )
            )

    def test_prepare_run_directory_removes_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            paths = prepare_run_directory(
                run_dir
            )

            temporary_path = (
                paths["metrics"]
                .with_suffix(
                    ".json.tmp"
                )
            )

            temporary_path.write_text(
                "temporary",
                encoding="utf-8",
            )

            prepare_run_directory(
                run_dir
            )

            self.assertFalse(
                temporary_path.exists()
            )


if __name__ == "__main__":
    unittest.main()
