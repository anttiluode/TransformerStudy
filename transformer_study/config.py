from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class ExperimentConfig:
    modulus: int = 8
    tape_len: int = 4
    demos: int = 3
    layers: int = 3
    d_model: int = 48
    heads: int = 4
    mlp_hidden: int = 96
    dropout: float = 0.0
    lr: float = 3e-4
    weight_decay: float = 0.0
    batch_size: int = 32
    steps: int = 2000
    grad_clip: float = 1.0
    model_seed: int = 17
    train_seed_root: int = 1000
    eval_seed_root: int = 2000
    map_fit_seed_root: int = 3000
    map_test_seed_root: int = 4000
    scramble_seed: int = 5000
    ridge_lambda: float = 1e-3
    eval_episodes: int = 192
    map_fit: int = 192
    map_test: int = 192
    trained_competence: float = 0.80
    novel_competence: float = 0.50
    correct_split_min_n: int = 20

    def to_dict(self) -> dict:
        return asdict(self)


def gate0_config() -> ExperimentConfig:
    return ExperimentConfig()


def gate1_config() -> ExperimentConfig:
    return replace(ExperimentConfig(), steps=8000)


def smoke_config() -> ExperimentConfig:
    return replace(
        ExperimentConfig(),
        layers=1,
        d_model=24,
        heads=4,
        mlp_hidden=48,
        batch_size=4,
        steps=8,
        eval_episodes=8,
        map_fit=8,
        map_test=8,
    )
