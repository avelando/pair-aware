import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.lifecycle import (
    RunLockedError,
    complete_run_attempt,
    fail_run_attempt,
    get_run_lock_path,
    get_run_status_path,
    read_run_status,
    run_lock,
    start_run_attempt,
)


class LifecycleTest(unittest.TestCase):
    def test_run_lock_records_active_and_released_state(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)

            with run_lock(
                run_dir,
                "experiment-1",
            ) as lock_metadata:
                self.assertTrue(
                    lock_metadata["active"]
                )

                stored = json.loads(
                    get_run_lock_path(
                        run_dir
                    ).read_text(
                        encoding="utf-8"
                    )
                )

                self.assertTrue(
                    stored["active"]
                )

                self.assertEqual(
                    stored["experiment_id"],
                    "experiment-1",
                )

            stored = json.loads(
                get_run_lock_path(
                    run_dir
                ).read_text(
                    encoding="utf-8"
                )
            )

            self.assertFalse(
                stored["active"]
            )

            self.assertIn(
                "released_at_utc",
                stored,
            )

    def test_second_lock_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)

            with run_lock(
                run_dir,
                "experiment-1",
            ):
                with self.assertRaisesRegex(
                    RunLockedError,
                    "Run is already locked",
                ):
                    with run_lock(
                        run_dir,
                        "experiment-1",
                    ):
                        pass

    def test_missing_status_returns_none(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(
                read_run_status(
                    Path(directory)
                )
            )

    def test_invalid_status_returns_none(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)

            get_run_status_path(
                run_dir
            ).write_text(
                "{invalid",
                encoding="utf-8",
            )

            self.assertIsNone(
                read_run_status(
                    run_dir
                )
            )

    def test_attempt_increments_for_same_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)

            first = start_run_attempt(
                run_dir,
                "experiment-1",
            )

            second = start_run_attempt(
                run_dir,
                "experiment-1",
            )

            self.assertEqual(
                first,
                1,
            )

            self.assertEqual(
                second,
                2,
            )

    def test_attempt_resets_for_new_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)

            start_run_attempt(
                run_dir,
                "experiment-1",
            )

            attempt = start_run_attempt(
                run_dir,
                "experiment-2",
            )

            self.assertEqual(
                attempt,
                1,
            )

    def test_complete_run_attempt_preserves_start_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)

            attempt = start_run_attempt(
                run_dir,
                "experiment-1",
            )

            running = read_run_status(
                run_dir
            )

            complete_run_attempt(
                run_dir=run_dir,
                experiment_id="experiment-1",
                attempt=attempt,
                duration_seconds=12.5,
            )

            completed = read_run_status(
                run_dir
            )

            self.assertEqual(
                completed["status"],
                "completed",
            )

            self.assertEqual(
                completed["attempt"],
                1,
            )

            self.assertEqual(
                completed["started_at_utc"],
                running["started_at_utc"],
            )

            self.assertEqual(
                completed["duration_seconds"],
                12.5,
            )

            self.assertIn(
                "completed_at_utc",
                completed,
            )

    def test_fail_run_attempt_records_error(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)

            attempt = start_run_attempt(
                run_dir,
                "experiment-1",
            )

            error = RuntimeError(
                "training failed"
            )

            fail_run_attempt(
                run_dir=run_dir,
                experiment_id="experiment-1",
                attempt=attempt,
                duration_seconds=3.5,
                error=error,
            )

            failed = read_run_status(
                run_dir
            )

            self.assertEqual(
                failed["status"],
                "failed",
            )

            self.assertEqual(
                failed["attempt"],
                1,
            )

            self.assertEqual(
                failed["error_type"],
                "RuntimeError",
            )

            self.assertEqual(
                failed["error_message"],
                "training failed",
            )

            self.assertEqual(
                failed["duration_seconds"],
                3.5,
            )

            self.assertIn(
                "failed_at_utc",
                failed,
            )


if __name__ == "__main__":
    unittest.main()