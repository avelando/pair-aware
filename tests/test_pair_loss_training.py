import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

import src.experiments.run_pair_aware as experiment_module
from src.config import PAIR_LOSS_WEIGHTS
from src.training.pair_aware import compute_pair_aware_loss


class PairLossTrainingTest(unittest.TestCase):
    def setUp(self):
        self.logits = torch.nn.Parameter(torch.tensor([[0.1, 0.7], [0.5, 0.2]]))
        self.batch = {
            "h_input_ids": torch.tensor([[1, 2]]),
            "n_input_ids": torch.tensor([[3, 4]]),
            "h_labels": torch.tensor([1]),
            "n_labels": torch.tensor([0]),
        }
        self.device = torch.device("cpu")

    def model(self, input_ids, labels):
        return SimpleNamespace(logits=self.logits, loss=F.cross_entropy(self.logits, labels))

    def test_zero_weight_matches_cross_entropy_and_its_gradient(self):
        result = compute_pair_aware_loss(self.model, self.batch, self.device, pair_loss_weight=0.0)
        expected = F.cross_entropy(self.logits, torch.tensor([1, 0]))
        actual_gradient = torch.autograd.grad(result["total_loss"], self.logits)[0]
        expected_gradient = torch.autograd.grad(expected, self.logits)[0]
        self.assertIs(result["total_loss"], result["classification_loss"])
        torch.testing.assert_close(result["total_loss"], expected)
        torch.testing.assert_close(actual_gradient, expected_gradient)

    def test_loss_and_gradient_use_requested_weight(self):
        for weight in PAIR_LOSS_WEIGHTS:
            with self.subTest(weight=weight):
                result = compute_pair_aware_loss(self.model, self.batch, self.device, pair_loss_weight=weight)
                classification = F.cross_entropy(self.logits, torch.tensor([1, 0]))
                scores = self.logits[:, 1] - self.logits[:, 0]
                expected = classification + weight * F.softplus(-(scores[0] - scores[1]))
                actual_gradient = torch.autograd.grad(result["total_loss"], self.logits)[0]
                expected_gradient = torch.autograd.grad(expected, self.logits)[0]
                torch.testing.assert_close(result["total_loss"], expected)
                torch.testing.assert_close(actual_gradient, expected_gradient)

    def test_invalid_weight_fails_before_model_execution(self):
        with self.assertRaises(ValueError):
            compute_pair_aware_loss(None, {}, self.device, pair_loss_weight=float("nan"))

    def test_experiment_passes_weight_to_runner_and_executor(self):
        with patch.object(experiment_module, "run_experiment", return_value={}) as run_mock:
            experiment_module.run_pair_aware("true_pair", 13, 40, pair_loss_weight=0.25)
        arguments = run_mock.call_args.kwargs
        self.assertEqual(arguments["pair_loss_weight"], 0.25)
        self.assertEqual(arguments["execute"].keywords["pair_loss_weight"], 0.25)

    def test_cli_accepts_zero_weight(self):
        arguments = [
            "run_pair_aware", "--method", "true_pair", "--split-seed", "13",
            "--model-seed", "40", "--pair-loss-weight", "0",
        ]
        with patch("sys.argv", arguments):
            parsed = experiment_module.parse_args()
        self.assertEqual(parsed.pair_loss_weight, 0.0)


if __name__ == "__main__":
    unittest.main()
