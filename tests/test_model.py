import pytest
import torch

from bigbird_enformer.models.enformer_plus import Enformer
from bigbird_enformer.utils.data import seq_indices_to_one_hot


def test_model_accepts_indices_and_one_hot(tiny_config_factory):
    config = tiny_config_factory()
    model = Enformer(config)
    model.eval()

    indices = torch.randint(0, 4, (1, 1024), dtype=torch.long)
    one_hot = seq_indices_to_one_hot(indices)
    with torch.no_grad():
        from_indices = model(indices)
        from_one_hot = model(one_hot)

    assert set(from_indices) == {"human", "mouse"}
    assert from_indices["human"].shape == (1, 4, 3)
    assert from_indices["mouse"].shape == (1, 4, 2)
    torch.testing.assert_close(from_indices["human"], from_one_hot["human"])
    torch.testing.assert_close(from_indices["mouse"], from_one_hot["mouse"])


def test_model_head_and_embedding_outputs(tiny_config_factory):
    model = Enformer(tiny_config_factory()).eval()
    sequence = torch.randint(0, 4, (1, 1024), dtype=torch.long)

    with torch.no_grad():
        outputs, embeddings = model(sequence, return_embeddings=True)
        human = model(sequence, head="human")
        shorter = model(sequence, head="human", target_length=2)

    assert embeddings.shape == (1, 4, 32)
    assert outputs["human"].shape == (1, 4, 3)
    assert human.shape == (1, 4, 3)
    assert shorter.shape == (1, 2, 3)


def test_model_rejects_target_longer_than_encoded_sequence(tiny_config_factory):
    model = Enformer(tiny_config_factory(target_length=5)).eval()
    sequence = torch.randint(0, 4, (1, 512), dtype=torch.long)

    with pytest.raises(ValueError, match="sequence length 4 < target length 5"):
        model(sequence)


def test_model_backward_is_finite(tiny_config_factory):
    model = Enformer(tiny_config_factory(target_length=2))
    sequence = torch.randint(0, 4, (2, 512), dtype=torch.long)

    loss = sum(value.mean() for value in model(sequence).values())
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.slow
def test_checkpointed_and_standard_forward_match(tiny_config_factory):
    plain = Enformer(tiny_config_factory(use_checkpointing=False)).eval()
    checkpointed = Enformer(tiny_config_factory(use_checkpointing=True)).eval()
    checkpointed.load_state_dict(plain.state_dict())
    sequence = torch.randint(0, 4, (1, 1024), dtype=torch.long)

    with torch.no_grad():
        expected = plain(sequence)
        actual = checkpointed(sequence)

    for organism in expected:
        torch.testing.assert_close(actual[organism], expected[organism])


@pytest.mark.slow
def test_full_length_block_sparse_smoke():
    from bigbird_enformer.utils.config import EnformerConfig

    config = EnformerConfig(
        dim=32,
        depth=1,
        heads=2,
        output_heads={"human": 5},
        target_length=10,
        attn_dim_key=8,
        attn_dim_value=8,
        block_size=64,
        attention_mode="block_sparse",
        dropout_rate=0.0,
        attn_dropout=0.0,
        dim_divisible_by=8,
    )
    model = Enformer(config).eval()
    sequence = torch.randint(0, 4, (1, 196_608), dtype=torch.long)

    with torch.no_grad():
        output = model(sequence)

    assert output["human"].shape == (1, 10, 5)
    assert torch.isfinite(output["human"]).all()
