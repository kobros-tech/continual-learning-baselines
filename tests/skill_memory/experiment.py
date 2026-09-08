import unittest

import torch
from torch import nn

from experiments.skill_memory import (
    SkillMemory,
    SkillMemoryPlugin,
    _interpolate_state_dicts,
    find_best_skill,
    find_best_weight_clone,
)


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, x):
        return self.net(x)


def _train_reference_model():
    torch.manual_seed(7)

    x = torch.tensor([
        [-2.0, -1.0],
        [-1.5, -2.0],
        [-2.0, -2.0],
        [-1.0, -1.5],
        [2.0, 1.0],
        [1.5, 2.0],
        [2.0, 2.0],
        [1.0, 1.5],
    ])

    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    model = TinyNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    criterion = nn.CrossEntropyLoss()

    for _ in range(100):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    eval_x = torch.tensor([
        [-1.8, -1.2],
        [-1.2, -1.8],
        [1.8, 1.2],
        [1.2, 1.8],
    ])

    eval_y = torch.tensor([0, 0, 1, 1])

    return model, eval_x, eval_y


def _state_statistics(state_dict):
    stats = {}

    for name, tensor in state_dict.items():
        if torch.is_floating_point(tensor):
            stats[name] = {
                "shape": tuple(tensor.shape),
                "mean": tensor.mean().item(),
                "std": tensor.std(unbiased=False).item(),
                "min": tensor.min().item(),
                "max": tensor.max().item(),
                "norm": tensor.norm().item(),
            }

    return stats


class SkillMemoryTest(unittest.TestCase):

    def test_store_keeps_independent_snapshot(self):
        memory = SkillMemory(max_skills=1)

        state = {
            "weight": torch.tensor([1.0, 2.0]),
        }

        slot = memory.allocate()
        memory.store(slot, state)

        state["weight"][0] = 99.0

        self.assertTrue(
            torch.equal(
                memory.state(slot)["weight"],
                torch.tensor([1.0, 2.0]),
            )
        )

    def test_memory_is_bounded(self):
        memory = SkillMemory(max_skills=1)

        slot = memory.allocate()
        memory.store(
            slot,
            {
                "weight": torch.tensor([1.0]),
            },
        )

        with self.assertRaises(RuntimeError):
            memory.allocate()

    def test_interpolation_preserves_real_model_namespace_and_endpoints(self):
        reference, _, _ = _train_reference_model()

        state = reference.state_dict()

        namespace = {
            name: tuple(tensor.shape)
            for name, tensor in state.items()
        }

        other = {
            name: tensor.clone()
            for name, tensor in state.items()
        }

        for name, tensor in other.items():
            if torch.is_floating_point(tensor):
                other[name] = tensor + 0.25

        at_zero = _interpolate_state_dicts(
            state,
            other,
            0.0,
        )

        at_one = _interpolate_state_dicts(
            state,
            other,
            1.0,
        )

        for name, tensor in state.items():
            self.assertEqual(
                tuple(tensor.shape),
                namespace[name],
            )

            self.assertTrue(
                torch.equal(
                    at_zero[name],
                    tensor,
                ),
                name,
            )

            self.assertTrue(
                torch.equal(
                    at_one[name],
                    other[name],
                ),
                name,
            )

    def test_clone_finds_real_trained_reference_initialization(self):
        """CLONE should recover a known optimum from empirical interpolation."""

        reference, eval_x, eval_y = _train_reference_model()

        reference_state = {
            name: tensor.detach().clone()
            for name, tensor in reference.state_dict().items()
        }

        state_a = {}
        state_b = {}

        for name, tensor in reference_state.items():
            if torch.is_floating_point(tensor):
                delta = torch.full_like(tensor, 0.40)

                state_a[name] = tensor - delta
                state_b[name] = tensor + delta
            else:
                state_a[name] = tensor.clone()
                state_b[name] = tensor.clone()

        midpoint = _interpolate_state_dicts(
            state_a,
            state_b,
            0.5,
        )

        for name, tensor in reference_state.items():
            if torch.is_floating_point(tensor):
                self.assertTrue(
                    torch.allclose(
                        midpoint[name],
                        tensor,
                        atol=1e-6,
                        rtol=1e-6,
                    ),
                    f"midpoint is not the reference state for {name}",
                )

        reference_stats = _state_statistics(reference_state)
        midpoint_stats = _state_statistics(midpoint)

        for name in reference_stats:
            for statistic in (
                "shape",
                "mean",
                "std",
                "min",
                "max",
                "norm",
            ):
                if statistic == "shape":
                    self.assertEqual(
                        reference_stats[name][statistic],
                        midpoint_stats[name][statistic],
                    )
                else:
                    self.assertAlmostEqual(
                        reference_stats[name][statistic],
                        midpoint_stats[name][statistic],
                        places=6,
                    )

        with torch.no_grad():
            reference_model = TinyNet()
            reference_model.load_state_dict(reference_state)
            reference_logits = reference_model(eval_x)

        def evaluate(candidate_state):
            model = TinyNet()
            model.load_state_dict(candidate_state)
            model.eval()

            with torch.no_grad():
                logits = model(eval_x)

                behavior_error = torch.mean(
                    (logits - reference_logits) ** 2
                ).item()

                compatibility = float(
                    torch.exp(
                        torch.tensor(-behavior_error)
                    ).item()
                )

                accuracy = float(
                    (
                        logits.argmax(dim=1) == eval_y
                    ).float().mean().item()
                )

            return compatibility, accuracy

        result = find_best_weight_clone(
            state_a,
            state_b,
            evaluate,
            score_skill=0,
            accuracy_skill=1,
            n_steps=21,
        )

        self.assertIsNotNone(result)

        self.assertAlmostEqual(
            result.alpha,
            0.5,
            places=6,
        )

        for name, tensor in reference_state.items():
            if torch.is_floating_point(tensor):
                self.assertTrue(
                    torch.allclose(
                        result.state_dict[name],
                        tensor,
                        atol=1e-6,
                        rtol=1e-6,
                    ),
                    f"CLONE result is not the reference state for {name}",
                )

        midpoint_compatibility, midpoint_accuracy = evaluate(midpoint)

        self.assertAlmostEqual(
            result.compatibility,
            midpoint_compatibility,
            places=6,
        )

        self.assertAlmostEqual(
            result.accuracy,
            midpoint_accuracy,
            places=6,
        )

    def test_find_best_weight_clone_finds_empirical_interior_optimum(self):
        """
        Test find_best_weight_clone directly with a synthetic evaluation
        landscape.

        The evaluator is deliberately artificial: the score is maximized
        at alpha=0.25 and accuracy at alpha=0.75. Their normalized average
        is maximized at alpha=0.50.

        This tests the clone-search logic itself without requiring model
        training or a real dataset.
        """

        state_a = {
            "weight": torch.tensor([0.0]),
        }

        state_b = {
            "weight": torch.tensor([1.0]),
        }

        def evaluate(candidate_state):
            alpha = candidate_state["weight"].item()

            score = 1.0 - (alpha - 0.25) ** 2
            accuracy = 1.0 - (alpha - 0.75) ** 2

            return score, accuracy

        result = find_best_weight_clone(
            state_a,
            state_b,
            evaluate,
            score_skill=10,
            accuracy_skill=20,
            n_steps=21,
        )

        self.assertIsNotNone(result)

        # 21 points gives:
        # 0.00, 0.05, ..., 0.50, ..., 1.00
        #
        # The synthetic combined objective is maximized at 0.50.
        self.assertAlmostEqual(
            result.alpha,
            0.50,
            places=6,
        )

        self.assertEqual(
            result.score_skill,
            10,
        )

        self.assertEqual(
            result.accuracy_skill,
            20,
        )

        self.assertEqual(
            result.direction,
            "score -> accuracy",
        )

        expected_state = _interpolate_state_dicts(
            state_a,
            state_b,
            0.50,
        )

        self.assertTrue(
            torch.allclose(
                result.state_dict["weight"],
                expected_state["weight"],
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_find_best_weight_clone_searches_both_directions(self):
        """
        The same physical interpolation path can be represented in either
        direction. Verify that the returned candidate is still the empirical
        optimum when the better representation is accuracy -> score.
        """

        score_state = {
            "weight": torch.tensor([1.0]),
        }

        accuracy_state = {
            "weight": torch.tensor([0.0]),
        }

        def evaluate(candidate_state):
            # Here alpha is represented by the actual weight value.
            x = candidate_state["weight"].item()

            score = 1.0 - (x - 0.75) ** 2
            accuracy = 1.0 - (x - 0.25) ** 2

            return score, accuracy

        result = find_best_weight_clone(
            score_state,
            accuracy_state,
            evaluate,
            score_skill=1,
            accuracy_skill=2,
            n_steps=21,
        )

        self.assertIsNotNone(result)

        # The optimum in the physical parameter space is x=0.50.
        self.assertAlmostEqual(
            result.state_dict["weight"].item(),
            0.50,
            places=6,
        )

        self.assertEqual(
            result.score_skill,
            1,
        )

        self.assertEqual(
            result.accuracy_skill,
            2,
        )

    def test_find_best_weight_clone_rejects_invalid_number_of_steps(self):
        state_a = {
            "weight": torch.tensor([0.0]),
        }

        state_b = {
            "weight": torch.tensor([1.0]),
        }

        def evaluate(candidate_state):
            return 1.0, 1.0

        with self.assertRaises(ValueError):
            find_best_weight_clone(
                state_a,
                state_b,
                evaluate,
                score_skill=0,
                accuracy_skill=1,
                n_steps=1,
            )

    def test_find_best_skill_requires_dynamic_agreement(self):
        results = [
            {
                "skill": 0,
                "old_accuracy": 0.90,
                "new_accuracy": 0.80,
                "new_score": 0.80,
                "chance": 0.50,
            },
            {
                "skill": 1,
                "old_accuracy": 0.90,
                "new_accuracy": 0.60,
                "new_score": 0.60,
                "chance": 0.50,
            },
            {
                "skill": 2,
                "old_accuracy": 0.90,
                "new_accuracy": 0.55,
                "new_score": 0.55,
                "chance": 0.50,
            },
        ]

        result = find_best_skill(
            results,
            forgetting_margin=0.05,
        )

        self.assertEqual(
            result["skill"],
            0,
        )

    def test_find_best_skill_rejects_disagreement_between_score_and_accuracy(self):
        results = [
            {
                "skill": 0,
                "old_accuracy": 0.90,
                "new_accuracy": 0.95,
                "new_score": 0.55,
                "chance": 0.50,
            },
            {
                "skill": 1,
                "old_accuracy": 0.90,
                "new_accuracy": 0.55,
                "new_score": 0.95,
                "chance": 0.50,
            },
            {
                "skill": 2,
                "old_accuracy": 0.90,
                "new_accuracy": 0.60,
                "new_score": 0.60,
                "chance": 0.50,
            },
        ]

        result = find_best_skill(
            results,
            forgetting_margin=0.05,
        )

        self.assertIsNone(result)

    def test_find_clone_requires_at_least_two_results(self):
        """
        _find_clone cannot form a clone from a single candidate skill.
        """
        plugin = SkillMemoryPlugin(
            max_skills=4,
            forgetting_margin=0.05,
            clone_steps=21,
            verbose=False,
        )

        skill = plugin.memory.allocate()

        state = {
            "net.0.weight": torch.zeros(4, 2),
            "net.0.bias": torch.zeros(4),
            "net.2.weight": torch.zeros(2, 4),
            "net.2.bias": torch.zeros(2),
        }

        plugin.memory.store(
            skill,
            state,
        )

        class Strategy:
            model = TinyNet()

        class Experience:
            dataset = torch.utils.data.TensorDataset(
                torch.zeros(2, 2),
                torch.tensor([0, 1]),
            )

        result = plugin._find_clone(
            Strategy(),
            Experience(),
            [
                {
                    "skill": skill,
                    "old_accuracy": 0.90,
                    "new_accuracy": 0.80,
                    "new_score": 0.80,
                    "chance": 0.50,
                },
            ],
        )

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
