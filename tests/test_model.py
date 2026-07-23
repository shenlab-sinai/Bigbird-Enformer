import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, "src"))


import torch
from bigbird_enformer.models.enformer_plus import Enformer
from bigbird_enformer.utils.config import EnformerConfig

def test_enformer_shapes():
    """
        testing code with a small model
    """
    config = EnformerConfig(
        dim=96,           
        depth=2,          
        heads=4,
        output_heads=dict(human=5),
        target_length=10,
        block_size=64,
        dim_divisible_by=8,
    )

    model = Enformer(config)
    model.eval()

    seq_len = 196608 
    x = torch.randint(0, 4, (1, seq_len))
    
    print(f"Input Shape: {x.shape}")

    with torch.no_grad():
        output = model(x)

    assert set(output) == {"human"}
    assert output["human"].shape == (1, 10, 5)

    print("Output Keys:", output.keys())
    print("Output Shape:", output['human'].shape)
    
if __name__ == "__main__":
    test_enformer_shapes()
