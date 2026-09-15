import os

os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path
import time

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.solver.gd import GradientSolver
import torch
from torchvision.transforms import v2 as transforms


# ---------------------------------------------------------------------------
# PATCH: stable-worldmodel 0.1.1 bug — GradientSolver.init_action() /
# prepare_init_action() leaves warm-start action tensor on CPU on the
# first planning call when init_action is None.
# ---------------------------------------------------------------------------
def _patched_init_action(self, n_envs, actions=None):
    actions = torch.randn(
        n_envs,
        self.num_samples,
        self.horizon,
        self.action_dim,
        generator=self.torch_gen,
        device=self.device,
        dtype=self.dtype,
    )

    if hasattr(self, "init") and self.init.shape == actions.shape:
        self.init.copy_(actions)
    else:
        if "init" in self._parameters:
            del self._parameters["init"]

        self.register_parameter(
            "init",
            torch.nn.Parameter(actions),
        )


GradientSolver.init_action = _patched_init_action


# ---------------------------------------------------------------------------
# LOSS TRACKING: Hook solver.step to capture planning losses per call
# ---------------------------------------------------------------------------
class LossTracker:
    def __init__(self):
        self.history = []

    def record(self, loss_val):
        self.history.append(loss_val)

    def reset(self):
        self.history.clear()


def install_loss_hook(solver: GradientSolver, tracker: LossTracker):
    """
    Wraps solver.step() to log the best sample's objective value at each
    planning execution.
    """
    orig_step = solver.step

    def wrapped_step(*args, **kwargs):
        actions = orig_step(*args, **kwargs)
        # solver.eval() tracks the evaluated objective/loss per trajectory sample
        if hasattr(solver, "last_loss") and solver.last_loss is not None:
            tracker.record(solver.last_loss.detach().cpu().item())
        elif hasattr(solver, "obj") and solver.obj is not None:
            # solver.obj shape: (batch_size, num_samples) -> min over samples
            min_loss = solver.obj.min().detach().cpu().item()
            tracker.record(min_loss)
        return actions

    solver.step = wrapped_step


def img_transform(cfg):
    """Transform images into the representation expected by the world model."""
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )


def get_episodes_length(dataset, episodes):
    """Return the length of each requested episode."""
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")

    lengths = [
        np.max(step_idx[episode_idx == ep_id]) + 1
        for ep_id in episodes
    ]
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    """Load the HDF5 dataset and cache the requested columns."""
    dataset_path = Path(
        cfg.get("cache_dir") or swm.data.utils.get_cache_dir()
    )
    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )


@hydra.main(
    version_base=None,
    config_path="./config/eval",
    config_name="pusht",
)
def run(cfg: DictConfig):
    """Run evaluation of the learned world model policy and log episode losses."""

    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block
        <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget

    world = swm.World(
        **cfg.world,
        image_shape=(224, 224),
    )

    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset

    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    ep_indices, _ = np.unique(
        stats_dataset.get_col_data(col_name),
        return_index=True,
    )

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue

        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)

        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = processor

    loss_tracker = LossTracker()
    policy_name = cfg.get("policy", "random")

    if policy_name != "random":
        ckpt_path = policy_name
        if not ckpt_path.endswith(".ckpt"):
            ckpt_path += ".ckpt"

        print(f"Loading local PyTorch model from {ckpt_path}...")
        model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = model.to(device)
        model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True

        config = swm.PlanConfig(**cfg.plan_config)

        solver = hydra.utils.instantiate(
            cfg.solver,
            model=model,
            device=device,
        )

        # Attach loss interceptor hook
        install_loss_hook(solver, loss_tracker)

        policy = swm.policy.WorldModelPolicy(
            solver=solver,
            config=config,
            process=process,
            transform=transform,
        )
    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if policy_name != "random"
        else Path(__file__).resolve().parent
    )

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {
        ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
    }

    all_row_ep_indices = dataset.get_col_data(col_name)
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in all_row_ep_indices]
    )

    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]

    print(f"{len(valid_indices)} valid starting points found for evaluation.")

    if len(valid_indices) < cfg.eval.num_eval:
        raise ValueError(
            f"Requested {cfg.eval.num_eval} evaluations, "
            f"but only {len(valid_indices)} valid starting steps exist."
        )

    g = np.random.default_rng(cfg.seed)
    chosen_sub_indices = g.choice(
        len(valid_indices),
        size=cfg.eval.num_eval,
        replace=False,
    )
    random_episode_indices = np.sort(valid_indices[chosen_sub_indices])

    selected_rows = dataset.get_row_data(random_episode_indices)
    eval_episodes = selected_rows[col_name]
    eval_start_idx = selected_rows["step_idx"]

    world.set_policy(policy)
    results_path.mkdir(parents=True, exist_ok=True)

    callables_cfg = cfg.eval.get("callables")
    callables = (
        OmegaConf.to_container(callables_cfg, resolve=True)
        if callables_cfg
        else None
    )

    # Callback to log losses per episode boundaries
    episode_final_losses = {}

    def per_episode_loss_callback(env, info):
        ep_id = info.get("episode_idx", len(episode_final_losses))
        if loss_tracker.history:
            episode_final_losses[int(ep_id)] = loss_tracker.history[-1]
            print(f"[Episode {ep_id}] Final Planning Loss: {loss_tracker.history[-1]:.6f}")
        else:
            episode_final_losses[int(ep_id)] = None

    if callables is None:
        callables = [per_episode_loss_callback]
    else:
        callables.append(per_episode_loss_callback)

    print("Starting evaluation...")
    start_time = time.time()

    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=callables,
        video=results_path,
    )

    end_time = time.time()

    print(metrics)
    print("\nFinal Planning Loss per Episode:")
    for ep_id, loss_val in episode_final_losses.items():
        loss_str = f"{loss_val:.6f}" if loss_val is not None else "N/A (Random/No steps)"
        print(f"  Episode {ep_id}: {loss_str}")

    output_file = results_path / cfg.output.filename
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("a") as f:
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))

        f.write("\n==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time:.4f} seconds\n")

        f.write("\n==== PER-EPISODE FINAL LOSS ====\n")
        for ep_id, loss_val in episode_final_losses.items():
            f.write(f"episode_{ep_id}_final_loss: {loss_val}\n")


if __name__ == "__main__":
    run()
