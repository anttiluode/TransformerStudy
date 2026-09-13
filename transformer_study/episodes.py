from dataclasses import dataclass
import numpy as np
import torch

from .config import ExperimentConfig
from .tasks import TRAIN_TASKS, TaskName, apply_task


@dataclass(frozen=True)
class Vocabulary:
    modulus: int

    @property
    def sep(self) -> int:
        return self.modulus

    @property
    def pair(self) -> int:
        return self.modulus + 1

    @property
    def bos(self) -> int:
        return self.modulus + 2

    @property
    def size(self) -> int:
        return self.modulus + 3


@dataclass
class Episode:
    tokens: list[int]
    loss_mask: list[int]
    task_name: TaskName
    query_input: list[int]
    query_target: list[int]
    query_sep_index: int
    query_output_start: int


@dataclass
class EpisodeBatch:
    tokens: torch.Tensor
    loss_mask: torch.Tensor
    task_names: list[TaskName]
    query_inputs: list[list[int]]
    query_targets: list[list[int]]
    query_sep_index: int


def episode_length(cfg: ExperimentConfig) -> int:
    return 1 + cfg.demos * (2 * cfg.tape_len + 2) + (2 * cfg.tape_len + 1)


def _draw_tape(cfg: ExperimentConfig, rng: np.random.Generator) -> list[int]:
    return rng.integers(0, cfg.modulus, size=cfg.tape_len).tolist()


def make_episode(
    task: TaskName,
    cfg: ExperimentConfig,
    rng: np.random.Generator,
    vocab: Vocabulary,
    query_input: list[int] | None = None,
) -> Episode:
    tokens = [vocab.bos]
    for _ in range(cfg.demos):
        x = _draw_tape(cfg, rng)
        y = apply_task(task, x, cfg.modulus)
        tokens += x + [vocab.sep] + y + [vocab.pair]
    qx = list(query_input) if query_input is not None else _draw_tape(cfg, rng)
    qy = apply_task(task, qx, cfg.modulus)
    tokens += qx + [vocab.sep]
    query_sep_index = len(tokens) - 1
    query_output_start = len(tokens)
    tokens += qy
    mask = [0] * len(tokens)
    mask[query_output_start : query_output_start + cfg.tape_len] = [1] * cfg.tape_len
    return Episode(tokens, mask, task, qx, qy, query_sep_index, query_output_start)


def make_eval_prompt(ep: Episode) -> list[int]:
    return ep.tokens[: ep.query_output_start]


def sample_training_batch(
    cfg: ExperimentConfig,
    vocab: Vocabulary,
    step: int,
    batch_size: int | None = None,
) -> EpisodeBatch:
    batch_size = cfg.batch_size if batch_size is None else batch_size
    episodes: list[Episode] = []
    for row in range(batch_size):
        index = step * batch_size + row
        task = TRAIN_TASKS[index % len(TRAIN_TASKS)]
        rng = np.random.default_rng(cfg.train_seed_root + index)
        episodes.append(make_episode(task, cfg, rng, vocab))
    return EpisodeBatch(
        tokens=torch.tensor([e.tokens for e in episodes], dtype=torch.long),
        loss_mask=torch.tensor([e.loss_mask for e in episodes], dtype=torch.bool),
        task_names=[e.task_name for e in episodes],
        query_inputs=[e.query_input for e in episodes],
        query_targets=[e.query_target for e in episodes],
        query_sep_index=episodes[0].query_sep_index,
    )
