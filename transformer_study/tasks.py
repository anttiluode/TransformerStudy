from enum import Enum


class TaskName(str, Enum):
    COPY = "copy"
    REVERSE = "reverse"
    SORT = "sort"
    CUMSUM_MOD = "cumsum_mod"
    PREFIX_PARITY = "prefix_parity"
    SWAP_PAIRS = "swap_pairs"
    DELTA_MOD = "delta_mod"


TRAIN_TASKS = (
    TaskName.COPY,
    TaskName.REVERSE,
    TaskName.SORT,
    TaskName.CUMSUM_MOD,
    TaskName.PREFIX_PARITY,
    TaskName.SWAP_PAIRS,
)
HELD_OUT_TASK = TaskName.DELTA_MOD


def apply_task(task: TaskName, tape: list[int], modulus: int) -> list[int]:
    x = list(tape)
    if not x:
        return []
    if task is TaskName.COPY:
        return x
    if task is TaskName.REVERSE:
        return x[::-1]
    if task is TaskName.SORT:
        return sorted(x)
    if task is TaskName.CUMSUM_MOD:
        total, out = 0, []
        for value in x:
            total = (total + value) % modulus
            out.append(total)
        return out
    if task is TaskName.PREFIX_PARITY:
        total, out = 0, []
        for value in x:
            total = (total + (value & 1)) & 1
            out.append(total)
        return out
    if task is TaskName.SWAP_PAIRS:
        if len(x) % 2:
            raise ValueError("swap_pairs requires even tape length")
        out = x[:]
        for i in range(0, len(x), 2):
            out[i], out[i + 1] = x[i + 1], x[i]
        return out
    if task is TaskName.DELTA_MOD:
        return [x[0]] + [(x[i] - x[i - 1]) % modulus for i in range(1, len(x))]
    raise ValueError(f"unknown task: {task}")
