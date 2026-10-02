import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import src.experiments.progress as progress_module


class ProgressTest(unittest.TestCase):
    def create_tracker(
        self,
        path,
    ):
        return (
            progress_module.RunProgressTracker(
                path=path,
                method="instance_level",
                split_seed=13,
                model_seed=40,
                experiment_id="experiment-id",
                attempt=2,
                total_epochs=6,
                emit_console=False,
            )
        )

    def test_format_duration(self):
        self.assertEqual(
            progress_module.format_duration(
                3661
            ),
            "01:01:01",
        )

        self.assertEqual(
            progress_module.format_duration(
                None
            ),
            "--:--:--",
        )

    def test_read_progress_returns_none_for_missing_and_invalid_files(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(
                directory
            )

            self.assertIsNone(
                progress_module.read_progress(
                    run_dir
                )
            )

            progress_module.get_progress_path(
                run_dir
            ).write_text(
                "{invalid",
                encoding="utf-8",
            )

            self.assertIsNone(
                progress_module.read_progress(
                    run_dir
                )
            )

    def test_start_writes_initial_training_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "progress.json"
            )

            tracker = self.create_tracker(
                path
            )

            with patch.object(
                progress_module,
                "perf_counter",
                return_value=100.0,
            ):
                result = tracker.start()

            self.assertEqual(
                result["status"],
                "running",
            )

            self.assertEqual(
                result["phase"],
                "training",
            )

            self.assertEqual(
                result["epoch"],
                0,
            )

            self.assertEqual(
                result["total_epochs"],
                6,
            )

            self.assertTrue(
                path.is_file()
            )

    def test_epoch_update_computes_elapsed_and_eta(self):
        with tempfile.TemporaryDirectory() as directory:
            tracker = self.create_tracker(
                Path(directory)
                / "progress.json"
            )

            with patch.object(
                progress_module,
                "perf_counter",
                side_effect=[
                    100.0,
                    160.0,
                ],
            ):
                tracker.start()

                result = tracker.update_epoch(
                    epoch=1,
                    epoch_metrics={
                        "validation_f1_macro": 0.8,
                    },
                    best_epoch=1,
                    best_validation_f1_macro=0.8,
                    epochs_without_improvement=0,
                )

            self.assertEqual(
                result["elapsed_seconds"],
                60.0,
            )

            self.assertEqual(
                result["last_epoch_seconds"],
                60.0,
            )

            self.assertEqual(
                result["average_epoch_seconds"],
                60.0,
            )

            self.assertEqual(
                result["eta_seconds"],
                300.0,
            )

            self.assertAlmostEqual(
                result["progress_fraction"],
                1 / 6,
            )

    def test_epoch_update_rejects_non_increasing_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            tracker = self.create_tracker(
                Path(directory)
                / "progress.json"
            )

            with patch.object(
                progress_module,
                "perf_counter",
                side_effect=[
                    100.0,
                    160.0,
                ],
            ):
                tracker.start()

                tracker.update_epoch(
                    epoch=1,
                    epoch_metrics={},
                    best_epoch=1,
                    best_validation_f1_macro=0.8,
                    epochs_without_improvement=0,
                )

            with self.assertRaisesRegex(
                ValueError,
                "Epoch progress must increase",
            ):
                tracker.update_epoch(
                    epoch=1,
                    epoch_metrics={},
                    best_epoch=1,
                    best_validation_f1_macro=0.8,
                    epochs_without_improvement=0,
                )

    def test_evaluation_phase_clears_eta(self):
        with tempfile.TemporaryDirectory() as directory:
            tracker = self.create_tracker(
                Path(directory)
                / "progress.json"
            )

            with patch.object(
                progress_module,
                "perf_counter",
                side_effect=[
                    100.0,
                    160.0,
                    170.0,
                ],
            ):
                tracker.start()

                tracker.update_epoch(
                    epoch=1,
                    epoch_metrics={},
                    best_epoch=1,
                    best_validation_f1_macro=0.8,
                    epochs_without_improvement=0,
                )

                result = tracker.set_phase(
                    "evaluation"
                )

            self.assertEqual(
                result["phase"],
                "evaluation",
            )

            self.assertIsNone(
                result["eta_seconds"]
            )

    def test_complete_marks_early_stopping(self):
        with tempfile.TemporaryDirectory() as directory:
            tracker = self.create_tracker(
                Path(directory)
                / "progress.json"
            )

            with patch.object(
                progress_module,
                "perf_counter",
                side_effect=[
                    100.0,
                    160.0,
                    170.0,
                ],
            ):
                tracker.start()

                tracker.update_epoch(
                    epoch=1,
                    epoch_metrics={},
                    best_epoch=1,
                    best_validation_f1_macro=0.8,
                    epochs_without_improvement=0,
                )

                result = tracker.complete()

            self.assertEqual(
                result["status"],
                "completed",
            )

            self.assertEqual(
                result["phase"],
                "completed",
            )

            self.assertEqual(
                result["eta_seconds"],
                0.0,
            )

            self.assertTrue(
                result["stopped_early"]
            )

    def test_fail_records_error(self):
        with tempfile.TemporaryDirectory() as directory:
            tracker = self.create_tracker(
                Path(directory)
                / "progress.json"
            )

            with patch.object(
                progress_module,
                "perf_counter",
                side_effect=[
                    100.0,
                    105.0,
                ],
            ):
                tracker.start()

                result = tracker.fail(
                    RuntimeError(
                        "training failed"
                    )
                )

            self.assertEqual(
                result["status"],
                "failed",
            )

            self.assertEqual(
                result["error_type"],
                "RuntimeError",
            )

            self.assertEqual(
                result["error_message"],
                "training failed",
            )


if __name__ == "__main__":
    unittest.main()