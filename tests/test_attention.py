import pytest
import torch

from bigbird_enformer.layers.attention import (
    BigBirdAttention,
    BigBirdAttentionAblation,
    BigBirdCCREAttention,
    BigBirdCCREAttentionEinsum,
    BlockSparseAttention,
    FullAttention,
    FullAttentionEinsum,
    RelBigBirdCCREAttention,
    RelFullAttention,
    get_positional_embed,
)
from bigbird_enformer.models.enformer_plus import (
    ChunkGlobalTokenInjector,
    ChunkGlobalTokenRemover,
    SingleGlobalTokenInjector,
    SingleGlobalTokenRemover,
)


def _randomize_output_projection(layer):
    with torch.no_grad():
        torch.nn.init.normal_(layer.to_out.weight, std=0.1)
        if layer.to_out.bias is not None:
            torch.nn.init.normal_(layer.to_out.bias, std=0.1)


@pytest.mark.parametrize(
    ("factory", "length"),
    [
        (lambda: FullAttention(12, heads=2, dim_key=4, dim_value=4), 8),
        (lambda: FullAttentionEinsum(12, heads=2, dim_key=4, dim_value=4), 8),
        (
            lambda: BigBirdCCREAttention(
                12,
                heads=2,
                dim_key=4,
                dim_value=4,
                block_size=4,
            ),
            8,
        ),
        (
            lambda: BigBirdCCREAttentionEinsum(
                12,
                heads=2,
                dim_key=4,
                dim_value=4,
                block_size=4,
            ),
            8,
        ),
        (
            lambda: BigBirdAttention(
                12,
                heads=2,
                dim_key=4,
                dim_value=4,
                block_size=4,
            ),
            9,
        ),
        (
            lambda: BlockSparseAttention(
                12,
                heads=2,
                dim_key=4,
                dim_value=4,
                block_size=4,
            ),
            8,
        ),
        (
            lambda: BigBirdAttentionAblation(
                12,
                heads=2,
                dim_key=4,
                dim_value=4,
                block_size=4,
            ),
            8,
        ),
        (
            lambda: RelFullAttention(
                12,
                heads=2,
                dim_key=4,
                dim_value=4,
                num_rel_pos_features=6,
                max_seq_len=8,
                pos_dropout=0.0,
            ),
            8,
        ),
        (
            lambda: RelBigBirdCCREAttention(
                12,
                heads=2,
                dim_key=4,
                dim_value=4,
                block_size=4,
                num_rel_pos_features=6,
                max_seq_len=8,
                pos_dropout=0.0,
            ),
            8,
        ),
    ],
    ids=[
        "full-sdpa",
        "full-einsum",
        "ccre-sdpa",
        "ccre-einsum",
        "bigbird-global-token",
        "block-sparse",
        "bigbird-ablation",
        "relative-full",
        "relative-ccre",
    ],
)
def test_attention_shapes_and_gradients_are_finite(factory, length):
    layer = factory()
    _randomize_output_projection(layer)
    inputs = torch.randn(2, length, 12, requires_grad=True)

    output = layer(inputs)
    output.square().mean().backward()

    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_full_sdpa_matches_einsum():
    sdpa = FullAttention(12, heads=2, dim_key=4, dim_value=4, dropout=0.0).eval()
    einsum = FullAttentionEinsum(
        12,
        heads=2,
        dim_key=4,
        dim_value=4,
        dropout=0.0,
    ).eval()
    _randomize_output_projection(sdpa)
    einsum.load_state_dict(sdpa.state_dict())
    inputs = torch.randn(2, 8, 12)

    torch.testing.assert_close(sdpa(inputs), einsum(inputs), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "mask",
    [
        torch.zeros(2, 8, dtype=torch.bool),
        torch.tensor(
            [
                [True, False, False, False, True, False, False, False],
                [False, True, False, False, False, True, False, False],
            ]
        ),
    ],
    ids=["local-only", "equal-global-counts"],
)
def test_ccre_sdpa_matches_einsum(mask):
    sdpa = BigBirdCCREAttention(
        12,
        heads=2,
        dim_key=4,
        dim_value=4,
        block_size=4,
    ).eval()
    einsum = BigBirdCCREAttentionEinsum(
        12,
        heads=2,
        dim_key=4,
        dim_value=4,
        block_size=4,
    ).eval()
    _randomize_output_projection(sdpa)
    einsum.load_state_dict(sdpa.state_dict())
    inputs = torch.randn(2, 8, 12)

    torch.testing.assert_close(
        sdpa(inputs, is_global=mask),
        einsum(inputs, is_global=mask),
        rtol=1e-5,
        atol=1e-6,
    )


def test_ccre_attention_batches_fixed_topk_masks_correctly():
    layer = BigBirdCCREAttention(
        12,
        heads=2,
        dim_key=4,
        dim_value=4,
        block_size=4,
    ).eval()
    _randomize_output_projection(layer)
    inputs = torch.randn(2, 8, 12)
    mask = torch.tensor(
        [
            [True, False, False, False, True, False, False, False],
            [False, True, False, False, False, True, False, False],
        ]
    )

    batched = layer(inputs, is_global=mask)
    separate = torch.cat(
        [
            layer(inputs[index : index + 1], is_global=mask[index : index + 1])
            for index in range(2)
        ]
    )

    torch.testing.assert_close(batched, separate)


def test_attention_rejects_non_divisible_sequence_length():
    layer = BigBirdCCREAttention(
        12,
        heads=2,
        dim_key=4,
        dim_value=4,
        block_size=4,
    )
    with pytest.raises(AssertionError, match="must be divisible"):
        layer(torch.randn(1, 10, 12))


def test_positional_embedding_shape_and_validation():
    embedding = get_positional_embed(
        seq_len=8,
        feature_size=12,
        device=torch.device("cpu"),
    )
    assert embedding.shape == (15, 12)
    assert torch.isfinite(embedding).all()

    with pytest.raises(ValueError, match="must be divisible"):
        get_positional_embed(
            seq_len=8,
            feature_size=10,
            device=torch.device("cpu"),
        )


def test_global_token_injectors_and_removers_are_inverse():
    inputs = torch.randn(2, 8, 12)

    single_injected = SingleGlobalTokenInjector(12)(inputs)
    single_restored = SingleGlobalTokenRemover()(single_injected)
    assert single_injected.shape == (2, 9, 12)
    torch.testing.assert_close(single_restored, inputs)

    chunk_injected = ChunkGlobalTokenInjector(12, num_chunks=2)(inputs)
    chunk_restored = ChunkGlobalTokenRemover(num_chunks=2)(chunk_injected)
    assert chunk_injected.shape == (2, 10, 12)
    torch.testing.assert_close(chunk_restored, inputs)
