import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn.functional as F

from src.training.common import (
    _create_optimizer_parameter_groups,
    backward_and_step,
    create_grad_scaler,
    create_optimizer_and_scheduler,
    load_model_state,
    move_batch_to_device,
    save_model_state,
    uses_amp,
)
from src.training.instance_level import train_instance_level_model
from src.training.pair_aware import (
    compute_pair_aware_loss,
    train_pair_aware_model,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.norm = torch.nn.LayerNorm(2)

    def forward(self, input_ids, labels=None):
        logits = self.norm(
            self.linear(
                input_ids.float()
            )
        )

        loss = None

        if labels is not None:
            loss = F.cross_entropy(
                logits,
                labels,
            )

        return SimpleNamespace(
            logits=logits,
            loss=loss,
        )


class FixedPairModel(torch.nn.Module):
    def forward(
        self,
        input_ids,
        labels=None,
    ):
        logits = torch.tensor(
            [
                [0.0, 2.0],
                [1.0, 3.0],
                [2.0, 0.0],
                [3.0, 1.0],
            ],
            dtype=torch.float32,
            device=input_ids.device,
        )

        loss = torch.tensor(
            0.5,
            dtype=torch.float32,
            device=input_ids.device,
        )

        return SimpleNamespace(
            logits=logits,
            loss=loss,
        )


class FakeScaler:
    def __init__(
        self,
        scale_before,
        scale_after,
        skip_step,
    ):
        self.current_scale = scale_before
        self.scale_after = scale_after
        self.skip_step = skip_step

    def get_scale(self):
        return self.current_scale

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        if not self.skip_step:
            optimizer.step()

    def update(self):
        self.current_scale = self.scale_after


class LoaderWithLength:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length


class TrainingTest(unittest.TestCase):
    def test_amp_is_enabled_only_for_cuda(self):
        self.assertTrue(
            uses_amp(
                torch.device("cuda")
            )
        )

        self.assertFalse(
            uses_amp(
                torch.device("cpu")
            )
        )

    def test_cpu_grad_scaler_is_disabled(self):
        scaler = create_grad_scaler(
            torch.device("cpu")
        )

        self.assertFalse(
            scaler.is_enabled()
        )

    def test_move_batch_to_device_excludes_requested_keys(self):
        batch = {
            "input_ids": torch.tensor(
                [[1, 2]]
            ),
            "attention_mask": torch.tensor(
                [[1, 1]]
            ),
            "labels": torch.tensor([1]),
        }

        result = move_batch_to_device(
            batch,
            torch.device("cpu"),
            excluded_keys=("labels",),
        )

        self.assertEqual(
            set(result),
            {
                "input_ids",
                "attention_mask",
            },
        )

        self.assertEqual(
            result["input_ids"].device.type,
            "cpu",
        )

    def test_optimizer_groups_exclude_bias_and_layer_norm_from_decay(self):
        model = TinyModel()

        groups = (
            _create_optimizer_parameter_groups(
                model
            )
        )

        decay_ids = {
            id(parameter)
            for parameter in groups[0][
                "params"
            ]
        }

        no_decay_ids = {
            id(parameter)
            for parameter in groups[1][
                "params"
            ]
        }

        self.assertIn(
            id(model.linear.weight),
            decay_ids,
        )

        self.assertIn(
            id(model.linear.bias),
            no_decay_ids,
        )

        self.assertIn(
            id(model.norm.weight),
            no_decay_ids,
        )

        self.assertIn(
            id(model.norm.bias),
            no_decay_ids,
        )

    def test_optimizer_and_scheduler_use_expected_training_steps(self):
        model = TinyModel()
        scheduler = MagicMock()

        with patch(
            "src.training.common.NUM_EPOCHS",
            5,
        ), patch(
            "src.training.common.WARMUP_RATIO",
            0.2,
        ), patch(
            "src.training.common.get_scheduler",
            return_value=scheduler,
        ) as get_scheduler_mock:
            (
                optimizer,
                result_scheduler,
                total_steps,
                warmup_steps,
            ) = create_optimizer_and_scheduler(
                model,
                steps_per_epoch=10,
            )

        self.assertIsInstance(
            optimizer,
            torch.optim.AdamW,
        )

        self.assertIs(
            result_scheduler,
            scheduler,
        )

        self.assertEqual(
            total_steps,
            50,
        )

        self.assertEqual(
            warmup_steps,
            10,
        )

        get_scheduler_mock.assert_called_once_with(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=10,
            num_training_steps=50,
        )

    def test_optimizer_rejects_non_positive_steps_per_epoch(self):
        model = TinyModel()

        with self.assertRaisesRegex(
            ValueError,
            "steps_per_epoch must be greater than zero",
        ):
            create_optimizer_and_scheduler(
                model,
                steps_per_epoch=0,
            )

    def test_backward_step_advances_scheduler_when_step_succeeds(self):
        model = torch.nn.Linear(2, 1)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.1,
        )

        scheduler = MagicMock()

        scaler = FakeScaler(
            scale_before=8.0,
            scale_after=8.0,
            skip_step=False,
        )

        loss = model(
            torch.tensor(
                [[2.0, -1.0]]
            )
        ).sum()

        with patch(
            "src.training.common.torch.nn.utils.clip_grad_norm_",
            return_value=torch.tensor(2.5),
        ) as clip_mock:
            result = backward_and_step(
                loss=loss,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )

        self.assertFalse(
            result["step_skipped"]
        )

        self.assertEqual(
            result["gradient_norm"],
            2.5,
        )

        scheduler.step.assert_called_once_with()

        clip_mock.assert_called_once()

        clip_parameters, clip_max_norm = (
            clip_mock.call_args.args
        )

        self.assertEqual(
            {
                id(parameter)
                for parameter in clip_parameters
            },
            {
                id(parameter)
                for parameter in model.parameters()
            },
        )

        self.assertEqual(
            clip_max_norm,
            1.0,
        )

    def test_backward_step_does_not_advance_scheduler_after_overflow(self):
        model = torch.nn.Linear(2, 1)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.1,
        )

        scheduler = MagicMock()

        scaler = FakeScaler(
            scale_before=8.0,
            scale_after=4.0,
            skip_step=True,
        )

        loss = model(
            torch.tensor(
                [[2.0, -1.0]]
            )
        ).sum()

        result = backward_and_step(
            loss=loss,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )

        self.assertTrue(
            result["step_skipped"]
        )

        self.assertEqual(
            result["scale_before"],
            8.0,
        )

        self.assertEqual(
            result["scale_after"],
            4.0,
        )

        scheduler.step.assert_not_called()

    def test_model_state_round_trip_restores_parameters(self):
        model = torch.nn.Linear(2, 2)

        original_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = (
                Path(directory)
                / "checkpoint.pt"
            )

            save_model_state(
                model,
                checkpoint_path,
            )

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(10.0)

            load_model_state(
                model,
                checkpoint_path,
                torch.device("cpu"),
            )

        for key, value in model.state_dict().items():
            self.assertTrue(
                torch.equal(
                    value,
                    original_state[key],
                )
            )

    def test_missing_checkpoint_is_rejected(self):
        model = torch.nn.Linear(2, 2)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = (
                Path(directory)
                / "missing.pt"
            )

            with self.assertRaisesRegex(
                FileNotFoundError,
                "Checkpoint not found",
            ):
                load_model_state(
                    model,
                    checkpoint_path,
                    torch.device("cpu"),
                )

    def test_pair_aware_loss_matches_expected_margin_objective(self):
        model = FixedPairModel()

        batch = {
            "h_input_ids": torch.tensor(
                [
                    [1, 2],
                    [3, 4],
                ]
            ),
            "n_input_ids": torch.tensor(
                [
                    [5, 6],
                    [7, 8],
                ]
            ),
            "h_labels": torch.tensor(
                [1, 1]
            ),
            "n_labels": torch.tensor(
                [0, 0]
            ),
        }

        result = compute_pair_aware_loss(
            model,
            batch,
            torch.device("cpu"),
        )

        expected_pair_loss = float(
            F.softplus(
                torch.tensor(-4.0)
            ).item()
        )

        self.assertEqual(
            result["pair_batch_size"],
            2,
        )

        self.assertAlmostEqual(
            float(
                result[
                    "classification_loss"
                ].item()
            ),
            0.5,
        )

        self.assertAlmostEqual(
            float(
                result[
                    "mean_pair_margin"
                ].item()
            ),
            4.0,
        )

        self.assertAlmostEqual(
            float(
                result[
                    "pair_loss"
                ].item()
            ),
            expected_pair_loss,
        )

        self.assertAlmostEqual(
            float(
                result[
                    "total_loss"
                ].item()
            ),
            0.5
            + expected_pair_loss,
        )

    def test_pair_aware_loss_rejects_mismatched_input_fields(self):
        batch = {
            "h_input_ids": torch.tensor(
                [[1, 2]]
            ),
            "h_attention_mask": torch.tensor(
                [[1, 1]]
            ),
            "n_input_ids": torch.tensor(
                [[3, 4]]
            ),
            "h_labels": torch.tensor([1]),
            "n_labels": torch.tensor([0]),
        }

        with self.assertRaisesRegex(
            ValueError,
            "H and N inputs must contain the same model fields",
        ):
            compute_pair_aware_loss(
                FixedPairModel(),
                batch,
                torch.device("cpu"),
            )

    def test_pair_aware_loss_rejects_mismatched_label_shapes(self):
        batch = {
            "h_input_ids": torch.tensor(
                [
                    [1, 2],
                    [3, 4],
                ]
            ),
            "n_input_ids": torch.tensor(
                [
                    [5, 6],
                    [7, 8],
                ]
            ),
            "h_labels": torch.tensor(
                [1, 1]
            ),
            "n_labels": torch.tensor([0]),
        }

        with self.assertRaisesRegex(
            ValueError,
            "H and N labels must have the same shape",
        ):
            compute_pair_aware_loss(
                FixedPairModel(),
                batch,
                torch.device("cpu"),
            )

    def test_instance_training_uses_early_stopping_and_restores_best_checkpoint(self):
        model = TinyModel()
        train_loader = LoaderWithLength(2)
        validation_loader = object()
        optimizer = MagicMock()

        optimizer.param_groups = [
            {
                "lr": 2e-5,
            }
        ]

        scheduler = MagicMock()
        scaler = MagicMock()

        train_results = [
            {
                "loss": 0.7,
                "optimizer_steps": 2,
                "skipped_optimizer_steps": 0,
                "grad_scale": 32.0,
            },
            {
                "loss": 0.6,
                "optimizer_steps": 2,
                "skipped_optimizer_steps": 0,
                "grad_scale": 32.0,
            },
            {
                "loss": 0.5,
                "optimizer_steps": 2,
                "skipped_optimizer_steps": 0,
                "grad_scale": 32.0,
            },
        ]

        metric_results = [
            (
                {
                    "accuracy": 0.80,
                    "f1_macro": 0.80,
                },
                None,
                None,
                None,
            ),
            (
                {
                    "accuracy": 0.75,
                    "f1_macro": 0.70,
                },
                None,
                None,
                None,
            ),
            (
                {
                    "accuracy": 0.70,
                    "f1_macro": 0.60,
                },
                None,
                None,
                None,
            ),
        ]

        progress_callback = MagicMock()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = (
                Path(directory)
                / "checkpoint.pt"
            )

            with patch(
                "src.training.instance_level.NUM_EPOCHS",
                6,
            ), patch(
                "src.training.instance_level.EARLY_STOPPING_PATIENCE",
                2,
            ), patch(
                "src.training.instance_level.create_optimizer_and_scheduler",
                return_value=(
                    optimizer,
                    scheduler,
                    12,
                    1,
                ),
            ), patch(
                "src.training.instance_level.create_grad_scaler",
                return_value=scaler,
            ), patch(
                "src.training.instance_level.train_instance_epoch",
                side_effect=train_results,
            ), patch(
                "src.training.instance_level.predict_instances",
                return_value=(
                    np.zeros((2, 2)),
                    np.array([0, 1]),
                ),
            ), patch(
                "src.training.instance_level.evaluate_instance_logits",
                side_effect=metric_results,
            ), patch(
                "src.training.instance_level.save_model_state"
            ) as save_mock, patch(
                "src.training.instance_level.load_model_state"
            ) as load_mock:
                result = train_instance_level_model(
                    model=model,
                    train_loader=train_loader,
                    validation_loader=validation_loader,
                    checkpoint_path=checkpoint_path,
                    device=torch.device("cpu"),
                    progress_callback=progress_callback,
                )

        self.assertEqual(
            result["best_epoch"],
            1,
        )

        self.assertEqual(
            result["epochs_trained"],
            3,
        )

        self.assertEqual(
            result["training_steps"],
            6,
        )

        self.assertEqual(
            result["skipped_training_steps"],
            0,
        )

        self.assertEqual(
            result["planned_training_steps"],
            12,
        )

        self.assertEqual(
            result["warmup_steps"],
            1,
        )

        self.assertAlmostEqual(
            result[
                "best_validation_f1_macro"
            ],
            0.80,
        )

        self.assertEqual(
            progress_callback.call_count,
            3,
        )

        self.assertEqual(
            progress_callback.call_args_list[
                -1
            ].kwargs["epoch"],
            3,
        )

        self.assertEqual(
            progress_callback.call_args_list[
                -1
            ].kwargs["best_epoch"],
            1,
        )

        save_mock.assert_called_once()
        load_mock.assert_called_once()

    def test_pair_aware_training_selects_best_epoch_and_restores_checkpoint(self):
        model = TinyModel()
        train_loader = LoaderWithLength(2)
        validation_loader = object()
        optimizer = MagicMock()

        optimizer.param_groups = [
            {
                "lr": 2e-5,
            }
        ]

        scheduler = MagicMock()
        scaler = MagicMock()
        progress_callback = MagicMock()

        train_results = [
            {
                "total_loss": 0.9,
                "classification_loss": 0.6,
                "pair_loss": 0.3,
                "mean_pair_margin": 0.2,
                "optimizer_steps": 2,
                "skipped_optimizer_steps": 0,
                "grad_scale": 32.0,
            },
            {
                "total_loss": 0.8,
                "classification_loss": 0.5,
                "pair_loss": 0.3,
                "mean_pair_margin": 0.4,
                "optimizer_steps": 2,
                "skipped_optimizer_steps": 0,
                "grad_scale": 32.0,
            },
            {
                "total_loss": 0.7,
                "classification_loss": 0.5,
                "pair_loss": 0.2,
                "mean_pair_margin": 0.3,
                "optimizer_steps": 1,
                "skipped_optimizer_steps": 1,
                "grad_scale": 16.0,
            },
            {
                "total_loss": 0.6,
                "classification_loss": 0.4,
                "pair_loss": 0.2,
                "mean_pair_margin": 0.2,
                "optimizer_steps": 2,
                "skipped_optimizer_steps": 0,
                "grad_scale": 16.0,
            },
        ]

        metric_results = [
            (
                {
                    "accuracy": 0.60,
                    "f1_macro": 0.60,
                },
                None,
                None,
                None,
            ),
            (
                {
                    "accuracy": 0.75,
                    "f1_macro": 0.75,
                },
                None,
                None,
                None,
            ),
            (
                {
                    "accuracy": 0.70,
                    "f1_macro": 0.70,
                },
                None,
                None,
                None,
            ),
            (
                {
                    "accuracy": 0.65,
                    "f1_macro": 0.65,
                },
                None,
                None,
                None,
            ),
        ]

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = (
                Path(directory)
                / "checkpoint.pt"
            )

            with patch(
                "src.training.pair_aware.NUM_EPOCHS",
                6,
            ), patch(
                "src.training.pair_aware.EARLY_STOPPING_PATIENCE",
                2,
            ), patch(
                "src.training.pair_aware.create_optimizer_and_scheduler",
                return_value=(
                    optimizer,
                    scheduler,
                    12,
                    1,
                ),
            ), patch(
                "src.training.pair_aware.create_grad_scaler",
                return_value=scaler,
            ), patch(
                "src.training.pair_aware.train_pair_aware_epoch",
                side_effect=train_results,
            ), patch(
                "src.training.pair_aware.predict_instances",
                return_value=(
                    np.zeros((2, 2)),
                    np.array([0, 1]),
                ),
            ), patch(
                "src.training.pair_aware.evaluate_instance_logits",
                side_effect=metric_results,
            ), patch(
                "src.training.pair_aware.save_model_state"
            ) as save_mock, patch(
                "src.training.pair_aware.load_model_state"
            ) as load_mock:
                result = train_pair_aware_model(
                    model=model,
                    train_loader=train_loader,
                    validation_loader=validation_loader,
                    checkpoint_path=checkpoint_path,
                    device=torch.device("cpu"),
                    progress_callback=(
                        progress_callback
                    ),
                )

        self.assertEqual(
            result["best_epoch"],
            2,
        )

        self.assertEqual(
            result["epochs_trained"],
            4,
        )

        self.assertEqual(
            result["training_steps"],
            7,
        )

        self.assertEqual(
            result["skipped_training_steps"],
            1,
        )

        self.assertAlmostEqual(
            result[
                "best_validation_f1_macro"
            ],
            0.75,
        )

        self.assertEqual(
            len(result["history"]),
            4,
        )

        self.assertEqual(
            progress_callback.call_count,
            4,
        )

        self.assertEqual(
            progress_callback.call_args_list[
                -1
            ].kwargs["epoch"],
            4,
        )

        self.assertEqual(
            progress_callback.call_args_list[
                -1
            ].kwargs["best_epoch"],
            2,
        )

        self.assertEqual(
            save_mock.call_count,
            2,
        )

        load_mock.assert_called_once()

if __name__ == "__main__":
    unittest.main()