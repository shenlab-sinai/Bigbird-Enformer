from types import SimpleNamespace

import pytest
import torch

from bigbird_enformer.train_lightning import (
    BigBirdLightningModule,
    CCREClassifierHead,
    MeanPearsonCorrCoefPerChannel,
    poisson_loss,
)


def test_classifier_head_shape_and_topk_count():
    head = CCREClassifierHead(dim=12, hidden_dims=8, dropout=0.0).eval()
    inputs = torch.randn(3, 10, 12)

    logits = head(inputs)
    mask = head.topk_mask(inputs, k=4)

    assert logits.shape == (3, 10)
    assert mask.shape == (3, 10)
    assert mask.dtype == torch.bool
    assert mask.sum(dim=1).tolist() == [4, 4, 4]


def test_classifier_topk_clamps_out_of_range_values():
    head = CCREClassifierHead(dim=12, hidden_dims=8, dropout=0.0).eval()
    inputs = torch.randn(2, 5, 12)
    assert head.topk_mask(inputs, k=0).sum(dim=1).tolist() == [1, 1]
    assert head.topk_mask(inputs, k=99).sum(dim=1).tolist() == [5, 5]


def test_poisson_loss_matches_manual_calculation():
    prediction = torch.tensor([2.0, 4.0])
    target = torch.tensor([3.0, 1.0])
    expected = (prediction - target * prediction.log()).mean()
    torch.testing.assert_close(poisson_loss(prediction, target), expected)


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        ([1.0, 2.0, 3.0], 1.0),
        ([3.0, 2.0, 1.0], -1.0),
    ],
)
def test_pearson_metric_known_correlations(prediction, expected):
    target = torch.tensor([[[1.0], [2.0], [3.0]]])
    prediction_tensor = torch.tensor(prediction).view(1, 3, 1)
    metric = MeanPearsonCorrCoefPerChannel(n_channels=1)

    metric.update(prediction_tensor, target)

    torch.testing.assert_close(
        metric.compute(),
        torch.tensor(expected),
        rtol=1e-5,
        atol=1e-5,
    )


def test_lightning_module_rejects_invalid_classifier_options(tiny_config_factory):
    config = tiny_config_factory()
    with pytest.raises(ValueError, match="classifier_mode must be 'progressive'"):
        BigBirdLightningModule(config, classifier_mode="invalid")
    with pytest.raises(AssertionError, match="classifier_every"):
        BigBirdLightningModule(config, classifier_every=0)


def test_lightning_module_rejects_unsafe_cached_classifier_mode(
    tiny_config_factory,
):
    with pytest.raises(ValueError, match="not keyed by sequence identity"):
        BigBirdLightningModule(
            tiny_config_factory(),
            classifier_mode="cached",
        )


def test_resolved_ccre_selection_is_saved_in_checkpoint_hparams(
    tiny_config_factory,
):
    module = BigBirdLightningModule(
        tiny_config_factory(attention_mode="ccre_bigbird"),
        mean_ccre_k=137,
        ccre_condition="no_ctcf",
    )

    assert module.hparams["mean_ccre_k"] == 137
    assert module.hparams["ccre_condition"] == "no_ctcf"


def test_topk_routing_mixes_ground_truth_and_predicted_probabilities():
    predicted_probabilities = torch.tensor([[0.2, 0.9]])
    logits = torch.logit(predicted_probabilities)
    gt_mask = torch.tensor([[True, False]])

    mixed = BigBirdLightningModule._mix_topk_probabilities(
        logits,
        gt_mask,
        mix_ratio=0.25,
    )

    torch.testing.assert_close(mixed, torch.tensor([[0.8, 0.225]]))


def test_topk_routing_uses_predicted_probabilities_without_ground_truth():
    logits = torch.tensor([[-2.0, 0.0, 2.0]])

    mixed = BigBirdLightningModule._mix_topk_probabilities(
        logits,
        gt_mask=None,
        mix_ratio=1.0,
    )

    torch.testing.assert_close(mixed, logits.sigmoid())


def test_optimizer_separates_decay_and_no_decay_parameters(tiny_config_factory):
    module = BigBirdLightningModule(
        tiny_config_factory(),
        lr=1e-3,
        warmup_steps=2,
        weight_decay=0.2,
    )
    module._trainer = SimpleNamespace(max_steps=20)

    configuration = module.configure_optimizers()
    optimizer = configuration["optimizer"]

    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.2, 0.0]
    assert configuration["lr_scheduler"]["interval"] == "step"


def test_synthetic_training_step_returns_finite_loss(
    tiny_config_factory,
    monkeypatch,
):
    module = BigBirdLightningModule(
        tiny_config_factory(target_length=2),
        use_classifier=False,
        warmup_steps=2,
    )
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)
    batch = {
        "human": {
            "sequence": torch.randint(0, 4, (2, 512), dtype=torch.long),
            "target": torch.rand(2, 2, 3),
        },
        "mouse": {
            "sequence": torch.randint(0, 4, (2, 512), dtype=torch.long),
            "target": torch.rand(2, 2, 2),
        },
    }

    loss = module.training_step(batch, batch_idx=0)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
