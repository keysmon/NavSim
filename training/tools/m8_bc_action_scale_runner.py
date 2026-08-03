from __future__ import annotations

import mlagents.trainers
import numpy as np
from mlagents.torch_utils import torch
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.components.bc.module import BCModule
from mlagents.trainers.torch_entities.utils import ModelUtils


EXPECTED_MLAGENTS_VERSION = "1.1.0"
RAW_TO_ENVIRONMENT_SCALE = 3.0
PATCH_MARKER = "[M8BCActionScale] validated=mlagents-1.1.0 scale=3.0"


def scale_expert_continuous(expert: torch.Tensor) -> torch.Tensor:
    if expert is None or expert.numel() == 0:
        raise ValueError("continuous expert tensor must be non-empty")
    if not torch.is_floating_point(expert):
        raise TypeError("continuous expert tensor must use a floating dtype")
    return expert * RAW_TO_ENVIRONMENT_SCALE


def validate_installed_contract(version: str | None = None) -> None:
    actual = mlagents.trainers.__version__ if version is None else version
    if actual != EXPECTED_MLAGENTS_VERSION:
        raise RuntimeError(
            "M8 BC action-scale correction requires ML-Agents "
            f"{EXPECTED_MLAGENTS_VERSION}; found {actual}"
        )

    raw = torch.tensor([[-6.0, -3.0, 0.0, 3.0, 6.0]])
    environment = AgentAction(raw, None).to_action_tuple(clip=True).continuous
    np.testing.assert_allclose(
        environment,
        np.array([[-1.0, -1.0, 0.0, 1.0, 1.0]], dtype=np.float32),
        rtol=0,
        atol=1e-7,
    )


def corrected_behavioral_cloning_loss(
    module,
    selected_actions,
    log_probs,
    expert_actions,
) -> torch.Tensor:
    action_spec = module.policy.behavior_spec.action_spec
    loss = torch.tensor(
        0.0,
        dtype=selected_actions.continuous_tensor.dtype,
        device=selected_actions.continuous_tensor.device,
    )
    if action_spec.continuous_size > 0:
        selected = selected_actions.continuous_tensor
        expert = expert_actions.continuous_tensor
        if selected is None or expert is None or expert.numel() == 0:
            raise ValueError("continuous BC actions must be non-empty")
        if selected.shape != expert.shape:
            raise ValueError(
                "selected and expert continuous actions require matching shapes"
            )
        loss = loss + torch.nn.functional.mse_loss(
            selected,
            scale_expert_continuous(expert),
        )

    if action_spec.discrete_size > 0:
        one_hot_expert = ModelUtils.actions_to_onehot(
            expert_actions.discrete_tensor,
            action_spec.discrete_branches,
        )
        log_prob_branches = ModelUtils.break_into_branches(
            log_probs.all_discrete_tensor,
            action_spec.discrete_branches,
        )
        loss = loss + torch.mean(
            torch.stack(
                [
                    torch.sum(
                        -torch.nn.functional.log_softmax(branch, dim=1)
                        * expert_branch,
                        dim=1,
                    )
                    for branch, expert_branch in zip(
                        log_prob_branches,
                        one_hot_expert,
                    )
                ]
            )
        )
    return loss


def install_patch() -> None:
    validate_installed_contract()
    if getattr(BCModule, "_m8_action_scale_installed", False):
        return
    BCModule._behavioral_cloning_loss = corrected_behavioral_cloning_loss
    BCModule._m8_action_scale_installed = True


def run_training_cli() -> None:
    install_patch()
    print(PATCH_MARKER, flush=True)
    from mlagents.trainers.learn import main as mlagents_main

    mlagents_main()


if __name__ == "__main__":
    run_training_cli()
