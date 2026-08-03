import unittest
from types import SimpleNamespace

from mlagents.torch_utils import torch
from mlagents.trainers.torch_entities.action_log_probs import ActionLogProbs
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents_envs.base_env import ActionSpec, BehaviorSpec

from training.tools.m8_bc_action_scale_runner import (
    EXPECTED_MLAGENTS_VERSION,
    build_demo_audit,
    corrected_behavioral_cloning_loss,
    install_patch,
    replay_checkpoint,
    scale_expert_continuous,
    validate_installed_contract,
)


class ActionScaleTests(unittest.TestCase):
    def test_environment_targets_round_trip_through_pinned_action_transform(self):
        expert = torch.tensor(
            [[-1.0, -0.2], [0.0, 0.2], [1.0, 1.0]],
            dtype=torch.float32,
        )
        original = expert.clone()

        raw = scale_expert_continuous(expert)
        environment = torch.clamp(raw, -3.0, 3.0) / 3.0

        torch.testing.assert_close(
            raw,
            torch.tensor(
                [[-3.0, -0.6], [0.0, 0.6], [3.0, 3.0]],
                dtype=torch.float32,
            ),
        )
        torch.testing.assert_close(environment, expert)
        torch.testing.assert_close(expert, original)
        self.assertEqual(raw.dtype, expert.dtype)
        self.assertEqual(raw.device, expert.device)
        self.assertEqual(raw.shape, expert.shape)

    def test_empty_expert_tensor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            scale_expert_continuous(torch.empty((0, 2)))

    def test_installed_contract_accepts_pinned_version(self):
        validate_installed_contract(EXPECTED_MLAGENTS_VERSION)

    def test_installed_contract_rejects_other_version(self):
        with self.assertRaisesRegex(RuntimeError, "requires ML-Agents 1.1.0"):
            validate_installed_contract("0.0.0")


class HybridLossTests(unittest.TestCase):
    def setUp(self):
        behavior_spec = BehaviorSpec([], ActionSpec(2, (2,)))
        self.module = SimpleNamespace(
            policy=SimpleNamespace(behavior_spec=behavior_spec)
        )

    def test_corrected_loss_scales_continuous_target_and_preserves_discrete_ce(self):
        selected = AgentAction(
            torch.tensor([[3.0, 0.6]], requires_grad=True),
            [torch.tensor([0])],
        )
        expert = AgentAction(
            torch.tensor([[1.0, 0.2]]),
            [torch.tensor([0])],
        )
        branch_logits = torch.tensor([[2.0, -1.0]], requires_grad=True)
        log_probs = ActionLogProbs(None, [torch.tensor([0.0])], [branch_logits])

        loss = corrected_behavioral_cloning_loss(
            self.module, selected, log_probs, expert
        )
        expected_discrete = -torch.log_softmax(branch_logits, dim=1)[0, 0]

        torch.testing.assert_close(loss, expected_discrete)
        loss.backward()
        self.assertIsNotNone(selected.continuous_tensor.grad)
        self.assertIsNotNone(branch_logits.grad)

    def test_selected_and_expert_continuous_shapes_must_match(self):
        selected = AgentAction(
            torch.zeros((2, 2)),
            [torch.zeros(2, dtype=torch.long)],
        )
        expert = AgentAction(
            torch.zeros((2, 1)),
            [torch.zeros(2, dtype=torch.long)],
        )
        log_probs = ActionLogProbs(
            None,
            [torch.zeros(2)],
            [torch.zeros((2, 2))],
        )

        with self.assertRaisesRegex(ValueError, "matching shapes"):
            corrected_behavioral_cloning_loss(
                self.module, selected, log_probs, expert
            )

    def test_install_patch_is_idempotent(self):
        install_patch()
        install_patch()

    def test_corrected_loss_optimizes_raw_action_to_environment_equivalent(self):
        module = SimpleNamespace(
            policy=SimpleNamespace(
                behavior_spec=BehaviorSpec([], ActionSpec(1, ()))
            )
        )
        raw_forward = torch.nn.Parameter(torch.zeros((1, 1)))
        optimizer = torch.optim.SGD([raw_forward], lr=0.1)
        expert = AgentAction(torch.ones((1, 1)), None)
        log_probs = ActionLogProbs(None, None, None)

        for _ in range(100):
            optimizer.zero_grad()
            selected = AgentAction(raw_forward, None)
            loss = corrected_behavioral_cloning_loss(
                module, selected, log_probs, expert
            )
            loss.backward()
            optimizer.step()

        self.assertGreater(raw_forward.item(), 2.9)
        environment_forward = torch.clamp(raw_forward, -3.0, 3.0) / 3.0
        self.assertGreater(environment_forward.item(), 0.96)


class RealDemoAuditTests(unittest.TestCase):
    DEMO = "NavSim/Assets/Demonstrations/M8RampPush80.demo"

    def test_push80_demo_counts_and_actions_match_recorded_evidence(self):
        audit = build_demo_audit(self.DEMO)

        self.assertEqual(audit["processed_experiences"], 25344)
        self.assertEqual(audit["real_experiences"], 23808)
        self.assertEqual(audit["padding_experiences"], 1536)
        self.assertEqual(audit["terminal_markers"], 80)
        self.assertEqual(audit["episode_length_counts"]["23"], 1)
        self.assertEqual(
            sum(
                audit["episode_length_counts"].get(str(length), 0)
                for length in (300, 301, 302)
            ),
            79,
        )
        self.assertAlmostEqual(
            audit["forward_gt_0_9_fraction"],
            0.996682,
            places=6,
        )
        self.assertAlmostEqual(
            audit["zero_turn_fraction"],
            0.995632,
            places=6,
        )
        self.assertEqual(audit["jump_counts"], {"0": 23808})

    def test_old_checkpoint_reproduces_raw_environment_labels(self):
        result = replay_checkpoint(
            "results/m8_probeA_bc_push80_checkpoint_diag/"
            "RampAgent/RampAgent-149951.pt",
            self.DEMO,
        )

        self.assertLess(result["raw_environment_label_mse"], 0.01)
        self.assertGreater(result["raw_forward_mean"], 0.98)
        self.assertLess(result["raw_forward_mean"], 1.02)
        self.assertGreater(result["raw_forward_std"], 0.8)
        self.assertGreater(result["no_jump_probability"], 0.95)
        self.assertGreater(result["environment_forward_mean"], 0.32)
        self.assertLess(result["environment_forward_mean"], 0.34)


if __name__ == "__main__":
    unittest.main()
