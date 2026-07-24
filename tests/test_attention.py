import pytest
import torch

from bigbird_enformer.layers.attention import (
    _flex_ccre_attention,
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
from torch.nn.attention.flex_attention import flex_attention
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
        "ccre-flex",
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


def test_attention_value_dimension_defaults_to_model_width_per_head():
    full = FullAttention(12, heads=2, dim_key=4)
    ccre = BigBirdCCREAttention(12, heads=2, dim_key=4, block_size=4)

    for layer in (full, ccre):
        assert layer.dim_value == 6
        assert layer.inner_v == 12
        assert layer.to_v.out_features == 12
        assert layer.to_out.in_features == 12


def test_bigbird_global_token_attention_matches_dense_boundary_reference():
    torch.manual_seed(0)
    layer = BigBirdAttention(
        12,
        heads=2,
        dim_key=4,
        dim_value=4,
        block_size=2,
        dropout=0.0,
    ).eval()
    inputs = torch.randn(2, 9, 12)

    batch_size, total_length, _ = inputs.shape
    local_length = total_length - 1
    q = layer.to_q(inputs).view(
        batch_size, total_length, layer.heads, layer.dim_key
    ).transpose(1, 2)
    k = layer.to_k(inputs).view(
        batch_size, total_length, layer.heads, layer.dim_key
    ).transpose(1, 2)
    v = layer.to_v(inputs).view(
        batch_size, total_length, layer.heads, layer.dim_value
    ).transpose(1, 2)

    block_ids = torch.arange(local_length) // layer.block_size
    local = (block_ids[:, None] - block_ids[None, :]).abs() <= 1
    allowed = torch.zeros(total_length, total_length, dtype=torch.bool)
    allowed[0, :] = True
    allowed[1:, 0] = True
    allowed[1:, 1:] = local

    reference = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=allowed.view(1, 1, total_length, total_length),
        dropout_p=0.0,
    )
    reference = reference.transpose(1, 2).contiguous().view(
        batch_size, total_length, layer.inner_v
    )
    reference = layer.to_out(reference)

    torch.testing.assert_close(
        layer(inputs),
        reference,
        rtol=1e-5,
        atol=1e-6,
    )


def test_bigbird_ablation_attention_matches_dense_boundary_reference():
    torch.manual_seed(0)
    layer = BigBirdAttentionAblation(
        12,
        heads=2,
        dim_key=4,
        dim_value=4,
        block_size=2,
        dropout=0.0,
    ).eval()
    inputs = torch.randn(2, 8, 12)

    batch_size, seq_len, _ = inputs.shape
    q = layer.to_q(inputs).view(
        batch_size, seq_len, layer.heads, layer.dim_key
    ).transpose(1, 2)
    k = layer.to_k(inputs).view(
        batch_size, seq_len, layer.heads, layer.dim_key
    ).transpose(1, 2)
    v = layer.to_v(inputs).view(
        batch_size, seq_len, layer.heads, layer.dim_value
    ).transpose(1, 2)

    block_ids = torch.arange(seq_len) // layer.block_size
    allowed = (block_ids[:, None] - block_ids[None, :]).abs() <= 1
    reference = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=allowed.view(1, 1, seq_len, seq_len),
        dropout_p=0.0,
    )
    reference = reference.transpose(1, 2).contiguous().view(
        batch_size, seq_len, layer.inner_v
    )
    reference = layer.to_out(reference)

    torch.testing.assert_close(
        layer(inputs),
        reference,
        rtol=1e-5,
        atol=1e-6,
    )


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
def test_ccre_flex_matches_dense_einsum_reference(mask):
    flex = BigBirdCCREAttention(
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
    _randomize_output_projection(flex)
    einsum.load_state_dict(flex.state_dict())
    inputs = torch.randn(2, 8, 12)

    torch.testing.assert_close(
        flex(inputs, is_global=mask),
        einsum(inputs, is_global=mask),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BigBirdCCREAttention(
            12,
            heads=2,
            dim_key=4,
            dim_value=4,
            block_size=2,
        ),
        lambda: BigBirdCCREAttentionEinsum(
            12,
            heads=2,
            dim_key=4,
            dim_value=4,
            block_size=2,
        ),
    ],
    ids=["flex", "einsum"],
)
def test_ccre_attention_matches_dense_union_of_local_and_global_keys(factory):
    torch.manual_seed(0)
    layer = factory().eval()
    _randomize_output_projection(layer)
    inputs = torch.randn(2, 8, 12)
    is_global = torch.tensor(
        [
            [False, True, False, False, False, False, False, False],
            [False, False, False, True, False, False, False, True],
        ]
    )

    batch_size, seq_len, _ = inputs.shape
    q = layer.to_q(inputs).view(
        batch_size, seq_len, layer.heads, layer.dim_key
    ).transpose(1, 2)
    k = layer.to_k(inputs).view(
        batch_size, seq_len, layer.heads, layer.dim_key
    ).transpose(1, 2)
    v = layer.to_v(inputs).view(
        batch_size, seq_len, layer.heads, layer.dim_value
    ).transpose(1, 2)

    block_ids = torch.arange(seq_len) // layer.block_size
    local = (block_ids[:, None] - block_ids[None, :]).abs() <= 1
    allowed = (
        local.unsqueeze(0)
        | is_global[:, :, None]
        | is_global[:, None, :]
    )
    reference = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=allowed.unsqueeze(1),
        dropout_p=0.0,
    )
    reference = reference.transpose(1, 2).contiguous().view(
        batch_size, seq_len, layer.inner_v
    )
    reference = layer.to_out(reference)

    # Position 1 is both global and local to query 2, but the union has six keys.
    assert allowed[0, 2].sum().item() == 6
    torch.testing.assert_close(
        layer(inputs, is_global=is_global),
        reference,
        rtol=1e-5,
        atol=1e-6,
    )


def test_flex_ccre_kernel_matches_dense_union_reference():
    torch.manual_seed(0)
    layer = BigBirdCCREAttention(
        12,
        heads=2,
        dim_key=4,
        dim_value=4,
        block_size=2,
    ).eval()
    inputs = torch.randn(2, 8, 12)
    is_global = torch.tensor(
        [
            [False, True, False, False, False, False, False, False],
            [False, False, False, True, False, False, False, True],
        ]
    )

    batch_size, seq_len, _ = inputs.shape
    observed_sparsity = []

    def reference_flex_attention(q, k, v, *, block_mask):
        observed_sparsity.append(block_mask.sparsity())
        return flex_attention(q, k, v, block_mask=block_mask)

    with torch.no_grad():
        q = layer.to_q(inputs).view(
            batch_size, seq_len, layer.heads, layer.dim_key
        ).transpose(1, 2)
        k = layer.to_k(inputs).view(
            batch_size, seq_len, layer.heads, layer.dim_key
        ).transpose(1, 2)
        v = layer.to_v(inputs).view(
            batch_size, seq_len, layer.heads, layer.dim_value
        ).transpose(1, 2)

        block_ids = torch.arange(seq_len) // layer.block_size
        local = (block_ids[:, None] - block_ids[None, :]).abs() <= 1
        allowed = (
            local.unsqueeze(0)
            | is_global[:, :, None]
            | is_global[:, None, :]
        )
        reference = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=allowed.unsqueeze(1),
            dropout_p=0.0,
        )
        actual = _flex_ccre_attention(
            q,
            k,
            v,
            is_global,
            layer.block_size,
            reference_flex_attention,
        )

    assert observed_sparsity[0] > 0
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)


def test_ccre_attention_batches_variable_global_counts_correctly():
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
            [False, True, False, False, False, False, False, False],
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


def test_ccre_flex_rejects_attention_dropout():
    with pytest.raises(ValueError, match="does not support.*attention dropout"):
        BigBirdCCREAttention(
            12,
            heads=2,
            dim_key=4,
            dim_value=4,
            block_size=4,
            dropout=0.05,
        )


def test_ccre_sdpa_backend_supports_attention_dropout():
    layer = BigBirdCCREAttention(
        12,
        heads=2,
        dim_key=4,
        dim_value=4,
        block_size=4,
        dropout=0.05,
        backend="sdpa",
    )

    output = layer(torch.randn(2, 8, 12))

    assert output.shape == (2, 8, 12)
    assert torch.isfinite(output).all()


def test_ccre_flex_backend_requires_cuda():
    layer = BigBirdCCREAttention(
        12,
        heads=2,
        dim_key=4,
        dim_value=4,
        block_size=4,
        backend="flex",
    )

    with pytest.raises(RuntimeError, match="requires CUDA"):
        layer(torch.randn(2, 8, 12))


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
