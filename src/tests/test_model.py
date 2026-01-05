import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_folder = os.path.dirname(current_dir)      
project_root = os.path.dirname(src_folder)    
sys.path.insert(0, project_root)


import torch
from src.models.enformer_plus import Enformer
from src.utils.config import EnformerConfig

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
        dna_chunk_len=128,
        dim_divisible_by=8,
    )

    model = Enformer(config)
    model.eval()

    seq_len = 196608 
    x = torch.randint(0, 4, (1, seq_len))
    
    print(f"Input Shape: {x.shape}")

    with torch.no_grad():
        output = model(x)
        
    print("Output Keys:", output.keys())
    print("Output Shape:", output['human'].shape)
    
if __name__ == "__main__":
    test_enformer_shapes()