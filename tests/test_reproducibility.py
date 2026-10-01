import random
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.training.common import create_dataloader
from src.training.reproducibility import (
    create_torch_generator,
    set_model_seed,
)


class ReproducibilityTest(unittest.TestCase):
    def test_model_seed_reproduces_python_numpy_and_torch(self):
        with patch(
            "torch.cuda.is_available",
            return_value=False,
        ):
            set_model_seed(40)

            first = (
                random.random(),
                float(
                    np.random.random()
                ),
                float(
                    torch.rand(1).item()
                ),
            )

            set_model_seed(40)

            second = (
                random.random(),
                float(
                    np.random.random()
                ),
                float(
                    torch.rand(1).item()
                ),
            )

        self.assertEqual(
            first,
            second,
        )

        self.assertTrue(
            torch.backends.cudnn.deterministic
        )

        self.assertFalse(
            torch.backends.cudnn.benchmark
        )

    def test_torch_generator_reproduces_sequence(self):
        first_generator = (
            create_torch_generator(73)
        )

        second_generator = (
            create_torch_generator(73)
        )

        first = torch.randperm(
            20,
            generator=first_generator,
        )

        second = torch.randperm(
            20,
            generator=second_generator,
        )

        self.assertTrue(
            torch.equal(
                first,
                second,
            )
        )

    def test_different_torch_generator_seeds_change_sequence(self):
        first = torch.randperm(
            20,
            generator=(
                create_torch_generator(13)
            ),
        )

        second = torch.randperm(
            20,
            generator=(
                create_torch_generator(21)
            ),
        )

        self.assertFalse(
            torch.equal(
                first,
                second,
            )
        )

    def test_shuffled_dataloader_requires_seed(self):
        dataset = TensorDataset(
            torch.arange(10)
        )

        with self.assertRaisesRegex(
            ValueError,
            "A seed is required when shuffle=True",
        ):
            create_dataloader(
                dataset,
                batch_size=2,
                shuffle=True,
            )

    def test_shuffled_dataloader_reproduces_order(self):
        dataset = TensorDataset(
            torch.arange(20)
        )

        with patch(
            "torch.cuda.is_available",
            return_value=False,
        ):
            first_loader = create_dataloader(
                dataset,
                batch_size=4,
                shuffle=True,
                seed=101,
            )

            second_loader = create_dataloader(
                dataset,
                batch_size=4,
                shuffle=True,
                seed=101,
            )

        first_order = torch.cat(
            [
                batch[0]
                for batch in first_loader
            ]
        )

        second_order = torch.cat(
            [
                batch[0]
                for batch in second_loader
            ]
        )

        self.assertTrue(
            torch.equal(
                first_order,
                second_order,
            )
        )

    def test_different_dataloader_seeds_change_order(self):
        dataset = TensorDataset(
            torch.arange(20)
        )

        with patch(
            "torch.cuda.is_available",
            return_value=False,
        ):
            first_loader = create_dataloader(
                dataset,
                batch_size=4,
                shuffle=True,
                seed=13,
            )

            second_loader = create_dataloader(
                dataset,
                batch_size=4,
                shuffle=True,
                seed=21,
            )

        first_order = torch.cat(
            [
                batch[0]
                for batch in first_loader
            ]
        )

        second_order = torch.cat(
            [
                batch[0]
                for batch in second_loader
            ]
        )

        self.assertFalse(
            torch.equal(
                first_order,
                second_order,
            )
        )

    def test_unshuffled_dataloader_preserves_order_without_seed(self):
        dataset = TensorDataset(
            torch.arange(10)
        )

        with patch(
            "torch.cuda.is_available",
            return_value=False,
        ):
            dataloader = create_dataloader(
                dataset,
                batch_size=3,
                shuffle=False,
            )

        order = torch.cat(
            [
                batch[0]
                for batch in dataloader
            ]
        )

        self.assertTrue(
            torch.equal(
                order,
                torch.arange(10),
            )
        )


if __name__ == "__main__":
    unittest.main()