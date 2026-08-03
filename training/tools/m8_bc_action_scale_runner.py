from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import mlagents.trainers
import numpy as np
from mlagents.torch_utils import torch
from mlagents.trainers.buffer import BufferKey
from mlagents.trainers.demo_loader import demo_to_buffer
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.components.bc.module import BCModule
from mlagents.trainers.torch_entities.networks import SimpleActor
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.trajectory import ObsUtil


EXPECTED_MLAGENTS_VERSION = "1.1.0"
RAW_TO_ENVIRONMENT_SCALE = 3.0
PATCH_MARKER = "[M8BCActionScale] validated=mlagents-1.1.0 scale=3.0"


def scale_expert_continuous(expert: torch.Tensor) -> torch.Tensor:
    if expert is None or expert.numel() == 0:
        raise ValueError("continuous expert tensor must be non-empty")
    if not torch.is_floating_point(expert):
        raise TypeError("continuous expert tensor must use a floating dtype")
    return expert * RAW_TO_ENVIRONMENT_SCALE


def _real_mask(done: np.ndarray, sequence_length: int) -> np.ndarray:
    done = np.asarray(done).reshape(-1)
    mask = np.ones(done.shape[0], dtype=bool)
    for terminal in np.flatnonzero(done):
        block_end = min(
            ((terminal // sequence_length) + 1) * sequence_length,
            done.shape[0],
        )
        if np.any(done[terminal + 1 : block_end]):
            raise ValueError("terminal block contains a second terminal marker")
        mask[terminal + 1 : block_end] = False
    return mask


def build_demo_audit(path: str, sequence_length: int = 64) -> dict:
    _, buffer = demo_to_buffer(path, sequence_length)
    done = np.asarray(buffer[BufferKey.DONE]).reshape(-1)
    terminals = np.flatnonzero(done)
    if terminals.size != 80:
        raise ValueError(f"expected 80 terminal markers; found {terminals.size}")

    real_mask = _real_mask(done, sequence_length)
    continuous = np.asarray(buffer[BufferKey.CONTINUOUS_ACTION])[real_mask]
    discrete = np.asarray(buffer[BufferKey.DISCRETE_ACTION])[real_mask]

    episode_lengths = []
    episode_start = 0
    for terminal in terminals:
        episode_lengths.append(int(terminal - episode_start + 1))
        episode_start = ((terminal // sequence_length) + 1) * sequence_length

    jump_counts = Counter(int(value) for value in discrete.reshape(-1))
    processed = int(done.shape[0])
    real = int(np.count_nonzero(real_mask))
    return {
        "processed_experiences": processed,
        "real_experiences": real,
        "padding_experiences": processed - real,
        "terminal_markers": int(terminals.size),
        "episode_length_counts": {
            str(length): count
            for length, count in sorted(Counter(episode_lengths).items())
        },
        "forward_gt_0_9_fraction": float(np.mean(continuous[:, 0] > 0.9)),
        "zero_turn_fraction": float(np.mean(np.abs(continuous[:, 1]) <= 1e-5)),
        "jump_counts": {
            str(value): count for value, count in sorted(jump_counts.items())
        },
    }


def replay_checkpoint(
    path: str,
    demo_path: str,
    sequence_length: int = 64,
) -> dict:
    behavior_spec, buffer = demo_to_buffer(demo_path, sequence_length)
    network_settings = NetworkSettings(
        normalize=True,
        hidden_units=256,
        num_layers=2,
        memory=NetworkSettings.MemorySettings(
            sequence_length=sequence_length,
            memory_size=128,
        ),
    )
    actor = SimpleActor(
        behavior_spec.observation_specs,
        network_settings,
        behavior_spec.action_spec,
    )
    checkpoint = torch.load(path, map_location="cpu")
    actor.load_state_dict(checkpoint["Policy"])
    actor.eval()

    observations = [
        ModelUtils.list_to_tensor(obs)
        for obs in ObsUtil.from_buffer(
            buffer,
            len(behavior_spec.observation_specs),
        )
    ]
    processed = buffer.num_experiences
    if processed % sequence_length != 0:
        raise ValueError("processed demo must contain complete sequences")
    sequence_count = processed // sequence_length
    memories = torch.zeros(1, sequence_count, actor.memory_size)
    masks = torch.ones(
        processed,
        sum(behavior_spec.action_spec.discrete_branches),
    )

    with torch.no_grad():
        encoding, _ = actor.network_body(
            observations,
            memories=memories,
            sequence_length=sequence_length,
        )
        distributions = actor.action_model._get_dists(encoding, masks)

    if distributions.continuous is None or distributions.discrete is None:
        raise ValueError("checkpoint must use the M8 hybrid action specification")

    done = np.asarray(buffer[BufferKey.DONE]).reshape(-1)
    real_mask = _real_mask(done, sequence_length)
    raw_mean = distributions.continuous.mean.detach().cpu().numpy()[real_mask]
    raw_std = distributions.continuous.std.detach().cpu().numpy()[real_mask]
    expert = np.asarray(buffer[BufferKey.CONTINUOUS_ACTION])[real_mask]
    no_jump = (
        distributions.discrete[0]
        .probs.detach()
        .cpu()
        .numpy()[real_mask, 0]
    )
    environment_mean = np.clip(raw_mean, -3.0, 3.0) / 3.0

    return {
        "raw_environment_label_mse": float(np.mean((raw_mean - expert) ** 2)),
        "raw_scaled_target_mse": float(
            np.mean((raw_mean - expert * RAW_TO_ENVIRONMENT_SCALE) ** 2)
        ),
        "raw_forward_mean": float(np.mean(raw_mean[:, 0])),
        "raw_turn_mean": float(np.mean(raw_mean[:, 1])),
        "raw_forward_std": float(np.mean(raw_std[:, 0])),
        "raw_turn_std": float(np.mean(raw_std[:, 1])),
        "environment_forward_mean": float(np.mean(environment_mean[:, 0])),
        "environment_turn_mean": float(np.mean(environment_mean[:, 1])),
        "no_jump_probability": float(np.mean(no_jump)),
    }


def run_audit_cli(arguments: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="m8_bc_action_scale_runner.py --m8-audit")
    parser.add_argument("--demo", required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="<label>=<path>",
    )
    parser.add_argument("--output", required=True)
    options = parser.parse_args(arguments)

    checkpoints = {}
    for value in options.checkpoint:
        if "=" not in value:
            parser.error("--checkpoint requires <label>=<path>")
        label, checkpoint_path = value.split("=", 1)
        if not label or not checkpoint_path:
            parser.error("--checkpoint requires non-empty <label>=<path>")
        if label in checkpoints:
            parser.error(f"duplicate checkpoint label: {label}")
        checkpoints[label] = replay_checkpoint(checkpoint_path, options.demo)

    validate_installed_contract()
    audit = {
        "contract": {
            "mlagents_version": EXPECTED_MLAGENTS_VERSION,
            "raw_to_environment_scale": RAW_TO_ENVIRONMENT_SCALE,
        },
        "demo": build_demo_audit(options.demo),
        "checkpoints": checkpoints,
    }
    output = Path(options.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("[M8BCActionScale] offline-audit PASS", flush=True)


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
    if len(sys.argv) > 1 and sys.argv[1] == "--m8-audit":
        run_audit_cli(sys.argv[2:])
    else:
        run_training_cli()
