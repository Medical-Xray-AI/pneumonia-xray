import torch

from src.models.densenet import DenseNet121Binary


def test_binary_head_and_output_shape_offline():
    model = DenseNet121Binary(pretrained=False, dropout=0.2, freeze_strategy="head_only")
    model.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
    assert model.backbone.classifier[-1].out_features == 1


def test_head_only_freezes_features():
    model = DenseNet121Binary(pretrained=False, freeze_strategy="head_only")
    assert all(not p.requires_grad for p in model.backbone.features.parameters())
    assert all(p.requires_grad for p in model.backbone.classifier.parameters())


def test_last_block_partial_finetuning():
    model = DenseNet121Binary(pretrained=False, freeze_strategy="last_block")
    early_block = list(model.backbone.features.denseblock1.parameters())
    last_block = list(model.backbone.features.denseblock4.parameters())
    assert early_block and last_block
    assert all(not p.requires_grad for p in early_block)
    assert all(p.requires_grad for p in last_block)


def test_binary_head_receives_gradients():
    model = DenseNet121Binary(pretrained=False, freeze_strategy="head_only")
    model.train()
    x = torch.randn(2, 3, 64, 64)
    y = torch.tensor([0.0, 1.0])
    logits = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    grads = [p.grad for p in model.backbone.classifier.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)
