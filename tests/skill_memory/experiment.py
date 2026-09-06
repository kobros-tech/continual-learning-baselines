import unittest

import torch

from experiments.skill_memory import SkillMemory


class SkillMemoryTest(unittest.TestCase):

    def test_register_keeps_independent_snapshot(self):
        memory = SkillMemory(max_skills=1)
        state = {"weight": torch.tensor([1.0, 2.0])}
        memory.register("skill-0", state)

        state["weight"][0] = 99.0
        self.assertTrue(torch.equal(memory._records["skill-0"].state_dict["weight"], torch.tensor([1.0, 2.0])))

    def test_memory_is_bounded(self):
        memory = SkillMemory(max_skills=1)
        memory.register("skill-0", {"weight": torch.tensor([1.0])})
        with self.assertRaises(RuntimeError):
            memory.register("skill-1", {"weight": torch.tensor([2.0])})
