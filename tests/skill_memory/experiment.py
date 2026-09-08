import unittest
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from experiments.skill_memory import (
    SkillMemory,
    _interpolate_state_dicts,
    find_best_skill,
    find_best_weight_clone,
)


class TinyClassifier(nn.Module):
    """Small real model used to exercise the clone path without Avalanche."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.net(x)


def make_binary_dataset(seed=0, n_samples=240):
    generator = torch.Generator().manual_seed(seed)
    half = n_samples // 2

    class_0 = torch.randn(half, 2, generator=generator) * 0.55
    class_0[:, 0] -= 1.0

    class_1 = torch.randn(half, 2, generator=generator) * 0.55
    class_1[:, 0] += 1.0

    x = torch.cat([class_0, class_1], dim=0)
    y = torch.cat([
        torch.zeros(half, dtype=torch.long),
        torch.ones(half, dtype=torch.long),
    ])

    permutation = torch.randperm(n_samples, generator=generator)
    return x[permutation], y[permutation]


def train_reference_model(x, y, seed=0):
    torch.manual_seed(seed)
    model = TinyClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.08)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(180):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    return model


def state_namespace(state_dict):
    """Architecture/schema namespace: parameter name -> tensor shape."""
    return {
        name: tuple(tensor.shape)
        for name, tensor in state_dict.items()
    }


def floating_statistics(state_dict):
    """Empirical statistics measured from a real trained model state."""
    statistics = {}
    for name, tensor in state_dict.items():
        if not torch.is_floating_point(tensor):
            continue
        statistics[name] = {
            "shape": tuple(tensor.shape),
            "mean": float(tensor.mean().item()),
            "std": float(tensor.std(unbiased=False).item()),
            "min": float(tensor.min().item()),
            "max": float(tensor.max().item()),
            "norm": float(torch.linalg.vector_norm(tensor).item()),
        }
    return statistics


def perturb_state(state_dict, reference, scale=0.75):
    """Create a source skill by perturbing a copy of a real trained state."""
    torch.manual_seed(1234)
    result = {}

    for name, tensor in reference.items():
        if not torch.is_floating_point(tensor):
            result[name] = tensor.clone()
            continue

        std = tensor.std(unbiased=False)
        magnitude = torch.clamp(std, min=0.05)
        noise = torch.randn_like(tensor)
        result[name] = tensor + scale * magnitude * noise

    return result


def state_distance(state_a, state_b):
    distances = []
    for name, tensor_a in state_a.items():
        tensor_b = state_b[name]
        if torch.is_floating_point(tensor_a):
            distances.append(
                float(torch.linalg.vector_norm(
                    tensor_a.float() - tensor_b.float()
                ).item())
            )
    return float(np.sqrt(np.sum(np.square(distances))))


class SkillMemoryTest(unittest.TestCase):

    def test_store_keeps_independent_snapshot(self):
        memory = SkillMemory(max_skills=1)
        state = {"weight": torch.tensor([1.0, 2.0])}
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
        first = memory.allocate()
        memory.store(first, {"weight": torch.tensor([1.0])})
        with self.assertRaises(RuntimeError):
            memory.allocate()

    def test_interpolation_preserves_real_model_namespace_and_endpoints(self):
        torch.manual_seed(7)
        model_a = TinyClassifier()
        torch.manual_seed(8)
        model_b = TinyClassifier()
        state_a = deepcopy(model_a.state_dict())
        state_b = deepcopy(model_b.state_dict())

        clone = _interpolate_state_dicts(state_a, state_b, 0.5)

        self.assertEqual(state_namespace(clone), state_namespace(state_a))

        for name in state_a:
            if torch.is_floating_point(state_a[name]):
                self.assertTrue(
                    torch.allclose(
                        clone[name].float(),
                        0.5 * state_a[name].float()
                        + 0.5 * state_b[name].float(),
                    )
                )
            else:
                self.assertTrue(
                    torch.equal(clone[name], state_a[name])
                    or torch.equal(clone[name], state_b[name])
                )

        # Interpolation must not mutate either stored skill.
        self.assertEqual(state_namespace(state_a), state_namespace(model_a.state_dict()))
        self.assertEqual(state_namespace(state_b), state_namespace(model_b.state_dict()))

    def test_clone_finds_real_trained_reference_initialization(self):
        """
        Scientific regression test for CLONE:

        1. Train a real model on a simple dataset.
        2. Measure its empirical weight statistics and schema namespace.
        3. Build two independent source skills around that trained state.
        4. Let the actual clone search evaluate interpolated candidates on the
           held-out dataset.
        5. Because the midpoint is exactly the trained reference state, it is
           an empirical optimum for both loss/compatibility and accuracy.

        This checks behavior, not merely tensor arithmetic: the clone must
        recover a state with the reference model's measured statistics and
        predictive performance.
        """
        x, y = make_binary_dataset(seed=11)
        train_x, test_x = x[:160], x[160:]
        train_y, test_y = y[:160], y[160:]

        reference_model = train_reference_model(train_x, train_y, seed=21)
        reference_state = deepcopy(reference_model.state_dict())
        reference_stats = floating_statistics(reference_state)
        reference_namespace = state_namespace(reference_state)

        # The two source skills are copies around the same real trained model.
        # Their midpoint is exactly the independently trained reference state.
        score_state = perturb_state(reference_state, reference_state, scale=0.90)
        accuracy_state = perturb_state(reference_state, reference_state, scale=-0.90)
        midpoint = _interpolate_state_dicts(
            score_state,
            accuracy_state,
            0.5,
        )

        for name, tensor in reference_state.items():
            self.assertTrue(
                torch.equal(midpoint[name], tensor),
                msg=f"midpoint is not the reference state for {name}",
            )

        criterion = nn.CrossEntropyLoss()
        cache = {}

        def evaluate(candidate_state):
            key = tuple(
                tensor.detach().cpu().numpy().tobytes()
                for tensor in candidate_state.values()
                if torch.is_floating_point(tensor)
            )
            if key in cache:
                return cache[key]

            model = TinyClassifier()
            model.load_state_dict(candidate_state)
            model.eval()
            with torch.no_grad():
                logits = model(test_x)
                loss = float(criterion(logits, test_y).item())
                accuracy = float(
                    (logits.argmax(dim=1) == test_y)
                    .float()
                    .mean()
                    .item()
                )

            compatibility = float(np.exp(-loss))
            cache[key] = (compatibility, accuracy)
            return cache[key]

        endpoint_score = evaluate(score_state)
        endpoint_accuracy = evaluate(accuracy_state)
        reference_result = evaluate(reference_state)

        result = find_best_weight_clone(
            score_state,
            accuracy_state,
            evaluate,
            score_skill=0,
            accuracy_skill=1,
            n_steps=21,
        )

        self.assertIsNotNone(result)
        assert result is not None

        # The empirical reference midpoint must be at least as good as either
        # source on both metrics because it is the model trained on this task.
        self.assertGreaterEqual(
            reference_result[0] + 1e-7,
            max(endpoint_score[0], endpoint_accuracy[0]),
        )
        self.assertGreaterEqual(
            reference_result[1] + 1e-7,
            max(endpoint_score[1], endpoint_accuracy[1]),
        )

        # The grid contains alpha=0.5, so the selected clone should recover
        # the real trained reference state (possibly in either direction).
        self.assertAlmostEqual(result.alpha, 0.5, places=6)
        self.assertEqual(state_namespace(result.state_dict), reference_namespace)

        clone_stats = floating_statistics(result.state_dict)
        for name, stats in reference_stats.items():
            self.assertEqual(clone_stats[name]["shape"], stats["shape"])
            self.assertAlmostEqual(clone_stats[name]["mean"], stats["mean"], places=6)
            self.assertAlmostEqual(clone_stats[name]["std"], stats["std"], places=6)
            self.assertAlmostEqual(clone_stats[name]["norm"], stats["norm"], places=6)

        clone_metrics = evaluate(result.state_dict)
        self.assertAlmostEqual(clone_metrics[0], reference_result[0], places=6)
        self.assertAlmostEqual(clone_metrics[1], reference_result[1], places=6)
        self.assertLess(state_distance(result.state_dict, reference_state), 1e-6)

    def test_find_best_skill_requires_dynamic_agreement(self):
        common = {
            "old_accuracy": 0.90,
            "chance": 0.50,
        }

        results = [
            {
                **common,
                "skill": 0,
                "new_score": 0.82,
                "new_accuracy": 0.86,
            },
            {
                **common,
                "skill": 1,
                "new_score": 0.61,
                "new_accuracy": 0.64,
            },
        ]

        selected = find_best_skill(results, forgetting_margin=0.05)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["skill"], 0)

    def test_find_best_skill_rejects_disagreement_between_score_and_accuracy(self):
        common = {
            "old_accuracy": 0.90,
            "chance": 0.50,
        }

        results = [
            {
                **common,
                "skill": 0,
                "new_score": 0.90,
                "new_accuracy": 0.60,
            },
            {
                **common,
                "skill": 1,
                "new_score": 0.60,
                "new_accuracy": 0.90,
            },
        ]

        # Neither skill is simultaneously strongest by the dynamic score and
        # accuracy evidence, so reuse must not be forced by a fixed threshold.
        self.assertIsNone(
            find_best_skill(results, forgetting_margin=0.05)
        )


if __name__ == "__main__":
    unittest.main()
