import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def bigbird_block_sparse_attention(q, k, v, block_size, global_indices):
    """
    Sparse Attention with Periodic Global Tokens
    Implements Periodic Global + Local (Sliding Window).
    """
    b, h, n, d = q.shape
    num_blocks = n // block_size
    
    # View as Blocks: [b, h, num_blocks, block_size, head_dim]
    q_blocks = q.view(b, h, num_blocks, block_size, d)
    k_blocks = k.view(b, h, num_blocks, block_size, d)
    v_blocks = v.view(b, h, num_blocks, block_size, d)

    # --- Global Attention ---
    # Gather global blocks based on indices
    indices_tensor = torch.tensor(global_indices, device=q.device)
    
    # [b, h, num_globals, block_size, d]
    g_q_blocks = torch.index_select(q_blocks, 2, indices_tensor)
    g_k_blocks = torch.index_select(k_blocks, 2, indices_tensor)
    g_v_blocks = torch.index_select(v_blocks, 2, indices_tensor)

    # Global Queries (at indices) attending to all keys
    # Flatten globals to [b, h, num_globals*block_size, d]
    g_q_flat = g_q_blocks.view(b, h, -1, d)
    global_row_scores = torch.einsum("bhqd, bhkd -> bhqk", g_q_flat, k)
    
    # All Queries attending to Global Keys (at indices)
    g_k_flat = g_k_blocks.view(b, h, -1, d)
    global_col_scores = torch.einsum("bhnqd, bhgd -> bhnqg", q_blocks, g_k_flat)

    # --- Local Attention ---
    # Create a window of 3 blocks
    k_roll_left  = torch.roll(k_blocks, shifts=1, dims=2) # [-1, 0, 1, 2 ...]
    k_roll_right = torch.roll(k_blocks, shifts=-1, dims=2) # [1, 2, 3, ...]
    
    # Concatenate to make [b, h, num_blocks, 3*block_size, head_dim]
    k_window = torch.cat([k_roll_left, k_blocks, k_roll_right], dim=3)
    
    # Queries attend to the Key Window
    local_scores = torch.einsum("bhnqd, bhnkd -> bhnqk", q_blocks, k_window)

    scale = 1.0 / math.sqrt(d)
    
    # --- Compute Outputs ---
    
    # Global Rows Output
    attn_global_rows = F.softmax(global_row_scores * scale, dim=-1)
    out_global_rows = torch.matmul(attn_global_rows, v) # [b, h, num_globals*block_size, d]
    # Reshape back to blocks [b, h, num_globals, block_size, d]
    out_global_rows = out_global_rows.view(b, h, len(global_indices), block_size, d)

    # Local Output
    attn_local = F.softmax(local_scores * scale, dim=-1)
    
    # Construct Value Window (same as Key Window)
    v_roll_left  = torch.roll(v_blocks, shifts=1, dims=2) 
    v_roll_right = torch.roll(v_blocks, shifts=-1, dims=2)
    v_window = torch.cat([v_roll_left, v_blocks, v_roll_right], dim=3)
    
    out_local = torch.matmul(attn_local, v_window) 

    # Add contribution from Global Keys
    attn_global_cols = F.softmax(global_col_scores * scale, dim=-1)
    g_v_flat = g_v_blocks.view(b, h, -1, d)
    out_from_global_keys = torch.matmul(attn_global_cols, g_v_flat)
    
    # Reshape to match blocks
    out_from_global_keys = out_from_global_keys.view(b, h, num_blocks, block_size, d)
    
    # Combine Local + Global Context
    final_out = out_local + out_from_global_keys

    # Replace computed local output for Global Blocks with Global Row Output
    # (Because Global Row saw everything, it's more accurate than Local + Global Col)
    for i, idx in enumerate(global_indices):
        final_out[:, :, idx, :, :] = out_global_rows[:, :, i, :, :]

    # Flatten blocks back to sequence
    return final_out.view(b, h, n, d)


class BigBirdAttention(nn.Module):
    """ Main Attention class"""
    def __init__(self, d_model, num_heads, block_size=64):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.block_size = block_size
        self.head_dim = d_model // num_heads
        
        # Define the Linear Layers
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, global_indices=None):
        """ 
        global_indices: list of block indices that act as hubs.
        If None, defaults to [0, -1]
        """
        batch_size, seq_len, _ = x.shape
        
        # Default to First/Last if no indices provided
        if global_indices is None:
            num_blocks = seq_len // self.block_size
            global_indices = [0, num_blocks - 1]

        # Project Q, K, V
        # Shape: [Batch, Heads, Seq_Len, Head_Dim]
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Perform sparse attention
        context = bigbird_block_sparse_attention(
            q, k, v, 
            self.block_size, 
            global_indices
        )

        # Final Projection
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.out_proj(context)