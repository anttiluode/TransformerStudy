import torch

from transformer_study.config import smoke_config
from transformer_study.episodes import Vocabulary
from transformer_study.model import TinyTransformer, gelu_derivative


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


def test_gelu_derivative_matches_autograd():
    x = torch.tensor([-2.0, -0.2, 0.0, 0.7, 2.0], requires_grad=True)
    torch.nn.functional.gelu(x).sum().backward()
    assert torch.allclose(gelu_derivative(x.detach()), x.grad, atol=1e-6)


def test_diagnostics_do_not_change_logits():
    cfg = smoke_config()
    vocab = Vocabulary(cfg.modulus)
    torch.manual_seed(3)
    model = TinyTransformer(cfg, vocab.size).eval()
    x = torch.randint(0, vocab.size, (2, 12))
    plain = model(x)
    diag_logits, diagnostics = model(x, return_diagnostics=True)
    assert torch.allclose(plain, diag_logits, atol=1e-6)
    assert len(diagnostics) == cfg.layers
    for diag in diagnostics:
        assert diag["mlp_preactivation"].shape == (2, 12, cfg.mlp_hidden)
        assert diag["gelu_derivative"].shape == (2, 12, cfg.mlp_hidden)
        assert diag["attention_weights"].shape[:3] == (2, cfg.heads, 12)
        assert diag["attention_weights"].shape[3] == 12


def test_identity_residual_patch_is_noop():
    cfg = smoke_config()
    vocab = Vocabulary(cfg.modulus)
    torch.manual_seed(4)
    model = TinyTransformer(cfg, vocab.size).eval()
    x = torch.randint(0, vocab.size, (2, 12))
    logits, residuals = model(x, return_residuals=True)
    patch = {
        "layer": 1,
        "position": x.shape[1] - 1,
        "value": residuals[1][:, -1, :].clone(),
    }
    patched = model(x, residual_patch=patch)
    assert torch.allclose(logits, patched, atol=1e-6)


def test_all_ones_mlp_mask_is_noop():
    cfg = smoke_config()
    vocab = Vocabulary(cfg.modulus)
    torch.manual_seed(5)
    model = TinyTransformer(cfg, vocab.size).eval()
    x = torch.randint(0, vocab.size, (2, 12))
    mask = {0: torch.ones(cfg.mlp_hidden)}
    assert torch.allclose(model(x), model(x, mlp_masks=mask), atol=1e-6)
