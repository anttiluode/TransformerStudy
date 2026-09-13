from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import ExperimentConfig
from .episodes import Vocabulary, make_episode, make_eval_prompt
from .model import TinyTransformer
from .tasks import HELD_OUT_TASK, TRAIN_TASKS, TaskName
from .train import greedy_query_output


@dataclass
class ResidualBank:
    states: dict[int, dict[TaskName, np.ndarray]]
    query_inputs: np.ndarray
    targets: dict[TaskName, np.ndarray]
    correct: dict[TaskName, np.ndarray]

    @property
    def n(self) -> int:
        return int(len(self.query_inputs))


@torch.no_grad()
def extract_paired_bank(
    model: TinyTransformer,
    cfg: ExperimentConfig,
    vocab: Vocabulary,
    n: int,
    bank_seed_root: int,
    tasks: tuple[TaskName, ...] | list[TaskName] | None = None,
    device: str | torch.device = "cpu",
) -> ResidualBank:
    device = torch.device(device)
    tasks = list(TRAIN_TASKS) + [HELD_OUT_TASK] if tasks is None else list(tasks)
    states_lists: dict[int, dict[TaskName, list[np.ndarray]]] = {
        layer: {task: [] for task in tasks} for layer in range(cfg.layers + 1)
    }
    targets_lists: dict[TaskName, list[list[int]]] = {task: [] for task in tasks}
    correct_lists: dict[TaskName, list[bool]] = {task: [] for task in tasks}
    queries: list[list[int]] = []
    model.eval()

    for row in range(n):
        query_rng = np.random.default_rng(bank_seed_root + row)
        query = query_rng.integers(0, cfg.modulus, size=cfg.tape_len).tolist()
        queries.append(query)
        for task_index, task in enumerate(tasks):
            demo_seed = bank_seed_root + 1_000_000 + row * 10_000 + task_index * 101
            demo_rng = np.random.default_rng(demo_seed)
            episode = make_episode(task, cfg, demo_rng, vocab, query_input=query)
            prompt = make_eval_prompt(episode)
            x = torch.tensor([prompt], dtype=torch.long, device=device)
            _, residuals = model(x, return_residuals=True)
            for layer, residual in enumerate(residuals):
                states_lists[layer][task].append(
                    residual[0, -1].detach().cpu().numpy().astype(np.float64, copy=True)
                )
            prediction = greedy_query_output(
                model, prompt, vocab, cfg.tape_len, device=device
            )
            targets_lists[task].append(episode.query_target)
            correct_lists[task].append(prediction == episode.query_target)

    states = {
        layer: {task: np.stack(values, axis=0) for task, values in task_map.items()}
        for layer, task_map in states_lists.items()
    }
    targets = {task: np.asarray(values, dtype=np.int64) for task, values in targets_lists.items()}
    correct = {task: np.asarray(values, dtype=bool) for task, values in correct_lists.items()}
    return ResidualBank(
        states=states,
        query_inputs=np.asarray(queries, dtype=np.int64),
        targets=targets,
        correct=correct,
    )
