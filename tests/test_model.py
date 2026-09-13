import torch

from transformer_study.config import smoke_config
from transformer_study.episodes import Vocabulary
from transformer_study.model import TinyTransformer


def test_forward_shapes_and_residual_count():
    cfg = smoke_config()
    vocab = Vocabulary(cfg.modulus)
    model = TinyTransformer(cfg, vocab.size)
    x = torch.zeros((2, 12), dtype=torch.long)
    logits, residuals = model(x, return_residuals=True)
    assert logits.shape == (2, 12, vocab.size)
    assert len(residuals) == cfg.layers + 1
    assert all(r.shape == (2, 12, cfg.d_model) for r in residuals)


def test_eval_forward_is_deterministic():
    cfg = smoke_config()
    vocab = Vocabulary(cfg.modulus)
    torch.manual_seed(1)
    model = TinyTransformer(cfg, vocab.size).eval()
    x = torch.randint(0, vocab.size, (2, 12))
    assert torch.equal(model(x), model(x))


def test_future_token_cannot_change_past_logits():
    cfg = smoke_config()
    vocab = Vocabulary(cfg.modulus)
    torch.manual_seed(2)
    model = TinyTransformer(cfg, vocab.size).eval()
    a = torch.randint(0, vocab.size, (1, 10))
    b = a.clone(); b[0, 9] = (b[0, 9] + 1) % vocab.size
    assert torch.allclose(model(a)[:, :9], model(b)[:, :9], atol=1e-6)
