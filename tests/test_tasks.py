import pytest

from transformer_study.config import gate0_config
from transformer_study.tasks import HELD_OUT_TASK, TRAIN_TASKS, TaskName, apply_task


def test_gate0_constants_are_frozen():
    cfg = gate0_config()
    assert (cfg.modulus, cfg.tape_len, cfg.demos) == (8, 4, 3)
    assert (cfg.layers, cfg.d_model, cfg.heads, cfg.mlp_hidden) == (3, 48, 4, 96)
    assert (cfg.steps, cfg.batch_size, cfg.lr) == (2000, 32, 3e-4)
    assert (cfg.eval_episodes, cfg.map_fit, cfg.map_test) == (192, 192, 192)


def test_exact_algorithms():
    x = [5, 1, 7, 2]
    assert apply_task(TaskName.COPY, x, 8) == [5, 1, 7, 2]
    assert apply_task(TaskName.REVERSE, x, 8) == [2, 7, 1, 5]
    assert apply_task(TaskName.SORT, x, 8) == [1, 2, 5, 7]
    assert apply_task(TaskName.CUMSUM_MOD, x, 8) == [5, 6, 5, 7]
    assert apply_task(TaskName.PREFIX_PARITY, x, 8) == [1, 0, 1, 1]
    assert apply_task(TaskName.SWAP_PAIRS, x, 8) == [1, 5, 2, 7]
    assert apply_task(TaskName.DELTA_MOD, x, 8) == [5, 4, 6, 3]


def test_delta_mod_is_never_a_training_task():
    assert HELD_OUT_TASK is TaskName.DELTA_MOD
    assert HELD_OUT_TASK not in TRAIN_TASKS
    assert set(TRAIN_TASKS) == {
        TaskName.COPY,
        TaskName.REVERSE,
        TaskName.SORT,
        TaskName.CUMSUM_MOD,
        TaskName.PREFIX_PARITY,
        TaskName.SWAP_PAIRS,
    }


def test_swap_pairs_requires_even_length():
    with pytest.raises(ValueError, match="even"):
        apply_task(TaskName.SWAP_PAIRS, [1, 2, 3], 8)
