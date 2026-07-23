import os
from pathlib import Path

import pytest
import torch

from bigbird_enformer.models.enformer_plus import Enformer


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU is not available",
)


@pytest.mark.gpu
@requires_cuda
def test_cuda_forward_backward(tiny_config_factory):
    model = Enformer(tiny_config_factory(target_length=2)).cuda()
    sequence = torch.randint(0, 4, (2, 512), dtype=torch.long, device="cuda")

    output = model(sequence)["human"]
    output.mean().backward()

    assert output.is_cuda
    assert torch.isfinite(output).all()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


@pytest.mark.gpu
@requires_cuda
def test_cpu_and_cuda_outputs_agree(tiny_config_factory):
    cpu_model = Enformer(tiny_config_factory(target_length=2)).eval()
    cuda_model = Enformer(tiny_config_factory(target_length=2)).cuda().eval()
    cuda_model.load_state_dict(cpu_model.state_dict())
    sequence = torch.randint(0, 4, (1, 512), dtype=torch.long)

    with torch.no_grad():
        cpu_output = cpu_model(sequence)["human"]
        cuda_output = cuda_model(sequence.cuda())["human"].cpu()

    torch.testing.assert_close(cuda_output, cpu_output, rtol=1e-4, atol=1e-5)


@pytest.mark.gpu
@requires_cuda
def test_cuda_bfloat16_autocast(tiny_config_factory):
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")
    model = Enformer(tiny_config_factory(target_length=2)).cuda().eval()
    sequence = torch.randint(0, 4, (1, 512), dtype=torch.long, device="cuda")

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(sequence)["human"]

    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()


@pytest.mark.integration
def test_external_checkpoint_contains_state_dict():
    checkpoint_path = os.environ.get("ATLAS_TEST_CHECKPOINT")
    if not checkpoint_path:
        pytest.skip("set ATLAS_TEST_CHECKPOINT to a Lightning checkpoint")
    path = Path(checkpoint_path)
    if not path.is_file():
        pytest.fail(f"ATLAS_TEST_CHECKPOINT does not exist: {path}")

    checkpoint = torch.load(path, map_location="cpu")

    assert isinstance(checkpoint, dict)
    assert isinstance(checkpoint.get("state_dict"), dict)
    assert checkpoint["state_dict"]
