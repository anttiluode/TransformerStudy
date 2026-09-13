import numpy as np

from transformer_study.config import gate0_config
from transformer_study.episodes import Vocabulary, episode_length, make_episode, sample_training_batch
from transformer_study.tasks import HELD_OUT_TASK, TRAIN_TASKS, TaskName


def test_episode_has_fixed_length_and_only_query_targets():
    cfg = gate0_config()
    vocab = Vocabulary(cfg.modulus)
    ep = make_episode(TaskName.REVERSE, cfg, np.random.default_rng(123), vocab)
    assert len(ep.tokens) == episode_length(cfg)
    assert len(ep.loss_mask) == len(ep.tokens)
    assert sum(ep.loss_mask) == cfg.tape_len
    assert ep.query_sep_index < ep.query_output_start
    assert ep.query_output_start == ep.query_sep_index + 1
    assert ep.loss_mask[ep.query_output_start:ep.query_output_start + cfg.tape_len] == [1] * cfg.tape_len


def test_training_sampler_is_balanced_and_never_samples_delta():
    cfg = gate0_config()
    vocab = Vocabulary(cfg.modulus)
    tasks = []
    for step in range(12):
        batch = sample_training_batch(cfg, vocab, step=step, batch_size=6)
        tasks.extend(batch.task_names)
    assert HELD_OUT_TASK not in tasks
    assert set(tasks) == set(TRAIN_TASKS)


def test_eval_prompt_stops_at_query_separator():
    cfg = gate0_config()
    vocab = Vocabulary(cfg.modulus)
    ep = make_episode(TaskName.COPY, cfg, np.random.default_rng(3), vocab)
    prompt = ep.tokens[: ep.query_output_start]
    assert prompt[-1] == vocab.sep
    assert len(prompt) == ep.query_output_start
