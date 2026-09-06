from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from avalanche.models.dynamic_modules import IncrementalClassifier, avalanche_model_adaptation
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


@dataclass
class SkillRecord:
    name: str
    state_dict: dict[str, Tensor]
    metadata: dict[str, Any] = field(default_factory=dict)


class SkillMemory:
    """Bounded registry of immutable model-state snapshots."""

    def __init__(self, max_skills: int = 20):
        if max_skills < 1:
            raise ValueError("max_skills must be positive")
        self.max_skills = max_skills
        self._records: dict[str, SkillRecord] = {}

    def register(self, name: str, state_dict: Mapping[str, Tensor], metadata=None):
        if name not in self._records and len(self._records) >= self.max_skills:
            raise RuntimeError(f"skill memory is at capacity ({self.max_skills})")
        self._records[name] = SkillRecord(
            name=name,
            state_dict={k: v.detach().cpu().clone() for k, v in state_dict.items()},
            metadata=dict(metadata or {}),
        )

    def __len__(self):
        return len(self._records)

    def records(self):
        return list(self._records.values())

    def load_into(self, name: str, model: nn.Module):
        state_dict = self._records[name].state_dict
        _resize_incremental_classifiers(model, state_dict)
        model.load_state_dict(deepcopy(state_dict), strict=False)


def _resize_incremental_classifiers(model: nn.Module, state_dict: Mapping[str, Tensor]):
    for module_name, module in model.named_modules():
        if not isinstance(module, IncrementalClassifier):
            continue
        prefix = f"{module_name}." if module_name else ""
        weight = state_dict.get(f"{prefix}classifier.weight")
        if weight is None or weight.ndim != 2:
            continue
        if module.classifier.out_features == weight.shape[0]:
            continue
        device = module.classifier.weight.device
        dtype = module.classifier.weight.dtype
        module.classifier = nn.Linear(
            module.classifier.in_features, weight.shape[0]
        ).to(device=device, dtype=dtype)
        active_key = f"{prefix}active_units"
        if active_key in state_dict:
            module.active_units = state_dict[active_key].to(device=device).clone()


def _restore_initial_state(model: nn.Module, initial_state: Mapping[str, Tensor]):
    current = model.state_dict()
    for name, initial in initial_state.items():
        if name not in current:
            continue
        target = current[name]
        if target.shape == initial.shape:
            target.copy_(initial.to(device=target.device, dtype=target.dtype))
        elif name.endswith("classifier.weight") and target.ndim == 2:
            rows = min(target.shape[0], initial.shape[0])
            target[:rows].copy_(initial[:rows].to(device=target.device, dtype=target.dtype))
        elif name.endswith("classifier.bias") and target.ndim == 1:
            rows = min(target.shape[0], initial.shape[0])
            target[:rows].copy_(initial[:rows].to(device=target.device, dtype=target.dtype))
        elif name.endswith("active_units"):
            continue
        else:
            raise RuntimeError(f"Cannot restore initial state for {name}")


def _origin_experience(experience):
    return getattr(experience, "origin_experience", experience)


def _probe(experience, samples: int, batches: int, seed: int | None = None):
    if len(experience.dataset) == 0:
        raise RuntimeError("Cannot probe an empty experience")
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    loader = DataLoader(
        experience.dataset,
        batch_size=min(samples, len(experience.dataset)),
        shuffle=True,
        generator=generator,
    )
    xs, ys = [], []
    for i, batch in enumerate(loader):
        if i >= max(1, batches):
            break
        xs.append(batch[0])
        ys.append(batch[1])
    return torch.cat(xs), torch.cat(ys)


def _evaluate_state(model: nn.Module, experience, x: Tensor, y: Tensor):
    model = deepcopy(model)
    _resize_incremental_classifiers(model, model.state_dict())
    avalanche_model_adaptation(model, _origin_experience(experience))
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(x.to(device))
        y = y.to(device)
        loss = nn.functional.cross_entropy(logits, y)
        accuracy = (logits.argmax(dim=1) == y).float().mean()
    return float(loss.item()), float(torch.exp(-loss).item()), float(accuracy.item())


class SkillMemoryPlugin(SupervisedPlugin):
    """Experimental Skill Memory plugin with REUSE, CLONE and SCRATCH decisions."""

    REUSE, CLONE, SCRATCH = "reuse", "clone", "scratch"

    def __init__(
        self,
        *,
        max_skills=20,
        reuse_threshold=0.90,
        clone_threshold=0.30,
        forgetting_margin=0.05,
        probe_samples=64,
        probe_batches=5,
        probe_seed=0,
    ):
        super().__init__()
        if not 0 <= clone_threshold <= reuse_threshold <= 1:
            raise ValueError("require 0 <= clone_threshold <= reuse_threshold <= 1")
        self.memory = SkillMemory(max_skills=max_skills)
        self.reuse_threshold = reuse_threshold
        self.clone_threshold = clone_threshold
        self.forgetting_margin = forgetting_margin
        self.probe_samples = probe_samples
        self.probe_batches = probe_batches
        self.probe_seed = probe_seed
        self.last_decision = self.SCRATCH
        self.last_selected_skill = None
        self.last_compatibility_score = 0.0
        self._initial_state = None
        self._saved_train_epochs = None
        self._seen_experiences = []
        self._task_active = False

    def _reset_optimizer(self, strategy):
        if strategy.optimizer is None:
            return
        strategy.optimizer.state.clear()

    def _scratch(self, strategy):
        _restore_initial_state(strategy.model, self._initial_state)
        self._reset_optimizer(strategy)

    def _probe_current(self, experience, offset=0):
        return _probe(
            experience,
            samples=self.probe_samples,
            batches=self.probe_batches,
            seed=self.probe_seed + offset,
        )

    def _score_records(self, strategy, experience):
        x, y = self._probe_current(experience)
        results = []
        for i, record in enumerate(self.memory.records()):
            candidate = deepcopy(strategy.model)
            self.memory.load_into(record.name, candidate)
            new_loss, new_score, new_accuracy = _evaluate_state(candidate, experience, x, y)

            old_index = record.metadata.get("experience")
            if not isinstance(old_index, int) or not 0 <= old_index < len(self._seen_experiences):
                continue
            old_exp = self._seen_experiences[old_index]
            old_x, old_y = self._probe_current(old_exp, 100003 + i)
            old_loss, _, old_accuracy = _evaluate_state(candidate, old_exp, old_x, old_y)
            results.append({
                "record": record,
                "new_score": new_score,
                "new_accuracy": new_accuracy,
                "new_loss": new_loss,
                "old_loss": old_loss,
                "old_accuracy": old_accuracy,
            })
        return results

    def before_training_exp(self, strategy, **kwargs):
        experience = strategy.experience
        if self._task_active and not getattr(experience, "is_first_subexp", True):
            return
        self._task_active = True
        if self._initial_state is None:
            self._initial_state = {
                k: v.detach().cpu().clone() for k, v in strategy.model.state_dict().items()
            }

        self.last_decision = self.SCRATCH
        self.last_selected_skill = None
        self.last_compatibility_score = 0.0
        self._saved_train_epochs = None

        if not self.memory:
            self._scratch(strategy)
            return
        if not self._seen_experiences:
            self._scratch(strategy)
            return

        results = self._score_records(strategy, experience)
        if not results:
            self._scratch(strategy)
            return

        classifier = getattr(strategy.model, "classifier", None)
        chance = 1.0 / max(1, getattr(classifier, "out_features", 10))
        safe = [
            r for r in results
            if r["old_accuracy"] > chance + self.forgetting_margin
        ]
        best = max(safe or results, key=lambda r: (r["new_score"], r["new_accuracy"]))
        self.last_selected_skill = best["record"].name
        self.last_compatibility_score = best["new_score"]

        if best in safe and best["new_score"] >= self.reuse_threshold:
            decision = self.REUSE
        elif best in safe and best["new_score"] >= self.clone_threshold:
            decision = self.CLONE
        else:
            decision = self.SCRATCH

        if decision in (self.REUSE, self.CLONE):
            self.memory.load_into(best["record"].name, strategy.model)
            avalanche_model_adaptation(strategy.model, _origin_experience(experience))
            self._reset_optimizer(strategy)
            self.last_decision = decision
            if decision == self.REUSE:
                self._saved_train_epochs = strategy.train_epochs
                strategy.train_epochs = 0
        else:
            self._scratch(strategy)

    def after_training_exp(self, strategy, **kwargs):
        experience = strategy.experience
        if not getattr(experience, "is_last_subexp", True):
            return
        if self._saved_train_epochs is not None:
            strategy.train_epochs = self._saved_train_epochs
            self._saved_train_epochs = None

        if self.last_decision != self.REUSE:
            name = f"experience-{getattr(experience, 'current_experience', len(self.memory))}"
            self.memory.register(
                name,
                strategy.model.state_dict(),
                metadata={
                    "acquisition_decision": self.last_decision,
                    "selected_skill": self.last_selected_skill,
                    "compatibility_score": self.last_compatibility_score,
                    "probe_samples": self.probe_samples,
                    "probe_batches": self.probe_batches,
                    "probe_seed": self.probe_seed,
                    "experience": getattr(
                        _origin_experience(experience),
                        "current_experience",
                        experience.current_experience,
                    ),
                },
            )
        self._seen_experiences.append(_origin_experience(experience))
        self._task_active = False
