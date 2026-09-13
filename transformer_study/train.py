from dataclasses import dataclass
import random

import numpy as np
import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .episodes import Vocabulary, make_episode, make_eval_prompt, sample_training_batch
from .model import TinyTransformer
from .tasks import HELD_OUT_TASK, TRAIN_TASKS, TaskName


@dataclass
class BehaviorMetrics:
    exact_accuracy: float
    token_accuracy: float
    correctness: list[bool]
    predictions: list[list[int]]
    targets: list[list[int]]

    def to_dict(self) -> dict:
        return {
            "exact": float(self.exact_accuracy),
            "token": float(self.token_accuracy),
        }


def masked_next_token_loss(
    logits: torch.Tensor, tokens: torch.Tensor, token_mask: torch.Tensor
) -> torch.Tensor:
    pred = logits[:, :-1, :].reshape(-1, logits.size(-1))
    target = tokens[:, 1:].reshape(-1)
    keep = token_mask[:, 1:].reshape(-1)
    if not bool(keep.any()):
        raise ValueError("token mask selects no prediction targets")
    return F.cross_entropy(pred[keep], target[keep])


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    cfg: ExperimentConfig,
    vocab: Vocabulary,
    device: str | torch.device = "cpu",
) -> tuple[TinyTransformer, list[float]]:
    _set_seeds(cfg.model_seed)
    device = torch.device(device)
    model = TinyTransformer(cfg, vocab.size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    history: list[float] = []
    model.train()
    for step in range(cfg.steps):
        batch = sample_training_batch(cfg, vocab, step=step)
        tokens = batch.tokens.to(device)
        mask = batch.loss_mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)
        loss = masked_next_token_loss(logits, tokens, mask)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite training loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    model.eval()
    return model, history


@torch.no_grad()
def greedy_query_output(
    model: TinyTransformer,
    prompt: list[int],
    vocab: Vocabulary,
    output_len: int,
    device: str | torch.device = "cpu",
) -> list[int]:
    device = torch.device(device)
    tokens = list(prompt)
    model.eval()
    for _ in range(output_len):
        x = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = model(x)
        # Query answers are defined to be symbol tokens only. Restricting the
        # decoder to that known output grammar avoids treating separators as
        # semantic answers while leaving all symbol probabilities untouched.
        next_token = int(logits[0, -1, : vocab.modulus].argmax().item())
        if not 0 <= next_token < vocab.modulus:
            raise RuntimeError("decoder selected a structural token")
        tokens.append(next_token)
    return tokens[-output_len:]


def _task_index(task: TaskName) -> int:
    ordered = list(TRAIN_TASKS) + [HELD_OUT_TASK]
    return ordered.index(task)


@torch.no_grad()
def evaluate_task(
    model: TinyTransformer,
    task: TaskName,
    cfg: ExperimentConfig,
    vocab: Vocabulary,
    n: int | None = None,
    seed_root: int | None = None,
    device: str | torch.device = "cpu",
) -> BehaviorMetrics:
    n = cfg.eval_episodes if n is None else n
    base = cfg.eval_seed_root if seed_root is None else seed_root
    predictions: list[list[int]] = []
    targets: list[list[int]] = []
    correctness: list[bool] = []
    token_hits = 0
    total_tokens = 0
    task_offset = _task_index(task) * 100_000
    for i in range(n):
        rng = np.random.default_rng(base + task_offset + i)
        episode = make_episode(task, cfg, rng, vocab)
        pred = greedy_query_output(
            model, make_eval_prompt(episode), vocab, cfg.tape_len, device=device
        )
        predictions.append(pred)
        targets.append(episode.query_target)
        ok = pred == episode.query_target
        correctness.append(ok)
        token_hits += sum(int(a == b) for a, b in zip(pred, episode.query_target))
        total_tokens += cfg.tape_len
    return BehaviorMetrics(
        exact_accuracy=float(sum(correctness) / max(1, n)),
        token_accuracy=float(token_hits / max(1, total_tokens)),
        correctness=correctness,
        predictions=predictions,
        targets=targets,
    )
