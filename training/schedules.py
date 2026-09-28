import math
from typing import Callable

SCHEDULE_TYPES = ("constant", "linear", "cosine", "wsd", "wsd_linear")


def make_lr_lambda(
    schedule_type: str,
    total_steps: int,
    warmup_steps: int,
    decay_ratio: float = 0.2,
    min_lr_ratio: float = 0.0,
) -> Callable[[int], float]:
    """Return a multiplier-on-peak-LR function of the optimizer step.

    `constant`, `linear` and `cosine` reproduce the trainer's original schedules
    exactly (including the 0.1 floor of the latter two), so default runs are unchanged.

    `wsd` and `wsd_linear` are warmup-stable-decay: hold the peak LR, then decay over the
    final `decay_ratio` of training down to `min_lr_ratio`, following 1 - sqrt(progress)
    (`wsd`) or 1 - progress (`wsd_linear`).
    """
    if schedule_type not in SCHEDULE_TYPES:
        raise ValueError(f"Unknown schedule_type '{schedule_type}'. Choose one of: {', '.join(SCHEDULE_TYPES)}")

    def warm(step: int):
        return step / warmup_steps if step < warmup_steps else None

    if schedule_type == "cosine":
        def fn(step):
            w = warm(step)
            if w is not None:
                return w
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    elif schedule_type == "linear":
        def fn(step):
            w = warm(step)
            if w is not None:
                return w
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.1, 1.0 - progress)
    elif schedule_type == "constant":
        def fn(step):
            w = warm(step)
            return w if w is not None else 1.0
    else:  # wsd, wsd_linear
        decay_steps = max(1, int(total_steps * decay_ratio))
        decay_start = total_steps - decay_steps
        shape = math.sqrt if schedule_type == "wsd" else (lambda p: p)

        def fn(step):
            w = warm(step)
            if w is not None:
                return w
            if step < decay_start:
                return 1.0
            progress = min(1.0, (step - decay_start) / decay_steps)
            return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - shape(progress))

    return fn
