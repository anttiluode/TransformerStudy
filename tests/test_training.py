from dataclasses import replace

import torch

from transformer_study.config import smoke_config
from transformer_study.episodes import Vocabulary
from transformer_study.train import masked_next_token_loss, train_model


def test_masked_loss_ignores_unmasked_positions():
    logits = torch.zeros((1, 5, 4))
    tokens = torch.tensor([[0, 1, 2, 3, 0]])
    mask = torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.bool)
    loss1 = masked_next_token_loss(logits, tokens, mask)
    logits[:, 0:2, :] = 1000.0
    loss2 = masked_next_token_loss(logits, tokens, mask)
    assert torch.allclose(loss1, loss2)


def test_two_step_training_has_finite_loss():
    cfg = replace(smoke_config(), steps=2)
    vocab = Vocabulary(cfg.modulus)
    model, history = train_model(cfg, vocab)
    assert len(history) == 2
    assert all(torch.isfinite(torch.tensor(v)) for v in history)
    assert model.training is False
