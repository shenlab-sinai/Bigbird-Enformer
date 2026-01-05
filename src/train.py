import sys
import os
import argparse
import logging
import math
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.cuda.amp as amp

# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.models.enformer_plus import Enformer
from src.utils.config import EnformerConfig
from src.utils.data import TFRecordDataLoader

# --- configuration (Based on DeepMind Notebook) ---
DEMO_DIM = 1536 // 4 
SEQ_LENGTH = 131_072 
TARGET_LENGTH = 896
NUM_TARGETS_HUMAN = 5313
NUM_TARGETS_MOUSE = 1643

def setup_logging(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(save_dir, "training.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )

def get_scheduler(optimizer, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0
    return LambdaLR(optimizer, lr_lambda)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--organism", type=str, default="human", choices=["human", "mouse"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    return parser.parse_args()

def train(args):
    setup_logging(args.save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device} | Organism: {args.organism}")
    train_loader = TFRecordDataLoader(organism=args.organism, subset='train', batch_size=1)
    
    num_targets = NUM_TARGETS_HUMAN if args.organism == 'human' else NUM_TARGETS_MOUSE
    
    config = EnformerConfig(
        dim=DEMO_DIM,         
        depth=11,              
        heads=8,          
        output_heads={args.organism: num_targets},
        target_length=TARGET_LENGTH,
        
        # Helper logic for BigBird
        block_size=64,
        dna_chunk_len=128,
        dim_divisible_by=8 
    )
    
    model = Enformer(config).to(device)
    model.train()
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=0.0005)
    scheduler = get_scheduler(optimizer, args.warmup_steps)
    scaler = amp.GradScaler()

    for epoch in range(args.epochs):
        logging.info(f"--- Epoch {epoch+1} ---")
        
        for step in range(args.steps_per_epoch):
            try:
                seq, target = next(train_loader)
                seq = seq.to(device)    
                target = target.to(device) 
                
                optimizer.zero_grad()
                
                with amp.autocast():
                    outputs = model(seq)
                    pred = outputs[args.organism]
                    loss = torch.nn.functional.poisson_nll_loss(pred, target, log_input=False)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                if step % 10 == 0:
                    lr = scheduler.get_last_lr()[0]
                    logging.info(f"Step {step} | Loss: {loss.item():.4f} | LR: {lr:.2e}")
                    
            except StopIteration:
                break
                
        torch.save(model.state_dict(), f"{args.save_dir}/model_epoch_{epoch}.pt")

if __name__ == "__main__":
    train(parse_args())