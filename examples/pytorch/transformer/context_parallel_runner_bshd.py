# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Test context parallel integration."""
from utils import get_dummy_data_bshd, collect_gradients, DistributedConfig
from model import SimpleConfig, SimpleBSHDModel

from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import get_thd_batch_on_this_cp_rank
import random
import numpy as np
import torch
import torch.distributed as dist
import os
import torch.nn.functional as F
from torch.distributed.tensor.device_mesh import init_device_mesh


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

data = get_dummy_data_bshd()


print(data)

DISTRIBUTED_MODE = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
print(f"Running in distributed mode: {DISTRIBUTED_MODE}")

##############################################################################
# STEP 1: RUN CP=1 (BASELINE) - NO DISTRIBUTED TRAINING
##############################################################################
print("\n" + "="*50)
print("RUNNING CP=1 (BASELINE) - Single GPU, No Distributed")
print("="*50)

# Simple setup for CP=1 - no distributed training needed
device = torch.device("cuda:0")  # Use first GPU
torch.cuda.set_device(0)

# Create config with CP=1
config = SimpleConfig(
    micro_batch_size=1,
    max_seq_length=1024,
    num_hidden_layers=2,
    vocab_size=33,
    hidden_size=320,
    intermediate_size=1280,
    num_attention_heads=20,
    layer_norm_eps=1e-5,
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.1,
)

# Create model with deterministic initialization - no DDP wrapper needed
model_cp1 = SimpleBSHDModel(config)
model_cp1 = model_cp1.to(device)

# No context parallel setup needed since CP=1
# No DDP wrapper needed since we're running on single GPU

# Run the same test with CP=1 using same random seed
torch.manual_seed(42)  # Same seed as CP=2 for identical randomness
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

batch = get_dummy_data_bshd()
batch = {k: v.to(device, non_blocking=True).contiguous() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    output_cp1 = model_cp1(batch)

print(f"CP=1 output shape: {output_cp1.shape}")
logits_flat_cp1 = output_cp1.view(-1, config.vocab_size)
labels_flat = batch["labels"].view(-1)
valid_mask = labels_flat != -100
if valid_mask.any():
    loss_cp1 = F.cross_entropy(logits_flat_cp1[valid_mask], labels_flat[valid_mask], reduction="mean")
else:
    loss_cp1 = logits_flat_cp1.sum() * 0.0

print(f"CP=1 loss: {loss_cp1.item():.6f}")

# Compute gradients for CP=1
loss_cp1.backward()

target_layers = [
    'embedding',           # Embedding layers
    'transformer_layers.0',      # First transformer layer
    'transformer_layers.1',      # Second transformer layer  
    'linear'              # Language model head
]

grads_cp1 = collect_gradients(model_cp1, layer_patterns=target_layers, max_params=15)

print(f"CP=1 collected {len(grads_cp1)} parameter gradients")

# Save the model state for reuse in CP=2 (no distributed checks needed)
initial_state_dict = {k: v.cpu().clone() for k, v in model_cp1.state_dict().items()}
torch.save(initial_state_dict, '/tmp/bshd_initial_model_state.pt')
print("Model state saved for CP=2 reuse")

# Store CP=1 results (only what we need to test)
cp1_results = {
    'logits': output_cp1.clone().detach().cpu(),
    'loss': loss_cp1.clone().detach().cpu(),
    'grad_norms': {name: grad.norm().item() for name, grad in grads_cp1.items()},
    'grads': grads_cp1,
}

torch.save(cp1_results, '/tmp/bshd_cp1_results.pt')
torch.save(data, '/tmp/bshd_data.pt')
print(f"CP=1 results: logits {cp1_results['logits'].shape}, loss {cp1_results['loss'].item():.6f}, {len(cp1_results['grad_norms'])} grad norms")

# No process group to destroy for CP=1 (single GPU run)
print("CP=1 baseline completed successfully")


# Ok now let's see if we can run the same model with context parallel

##############################################################################
# STEP 2: RUN CP=2 (CONTEXT PARALLELISM)
##############################################################################

# Skip CP=2 if not in distributed mode, otherwise run it
if not DISTRIBUTED_MODE:
    print("\n" + "="*50)
    print("SKIPPING CP=2 - Not running in distributed mode")
    print("To test CP=2, run with: torchrun --nproc_per_node=2 test_lightweight_integration.py")
    print("="*50)
    
    # Create dummy CP=2 results for comparison (same as CP=1)
    cp2_results = {
        'logits': cp1_results['logits'].clone(),
        'loss': cp1_results['loss'].clone(),
        'grad_norms': cp1_results['grad_norms'].copy()
    }
    import sys; sys.exit(1)  # Exit with error code 1
    print("Created dummy CP=2 results (identical to CP=1) for testing comparison logic")

# If we're in distributed mode, the rest of the CP=2 code will run normally
if DISTRIBUTED_MODE:
    print("\n" + "="*50)
    print("RUNNING CP=2 (CONTEXT PARALLELISM)")
    print("="*50)

    cp_size = 2

    # Reinitialize process group for CP=2
    print("Reinitializing process group for CP=2...")
    dist.init_process_group(backend="nccl")
    dist_config = DistributedConfig()
    torch.cuda.set_device(dist_config.local_rank)
    print(f"Rank {dist_config.rank}: Process group reinitialized")

    print("Creating device mesh...")
    device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(1, cp_size, 1),
            mesh_dim_names=("fsdp", "cp", "tp"),
        )
    device = torch.device(f"cuda:{dist_config.local_rank}")
    print(f"Rank {dist_config.rank}: Device mesh created")

    # Create model with same initialization as CP=1
    print("Creating model...")
    model = SimpleBSHDModel(config)

    # Load the same initial state dict from CP=1 run
    print("Loading saved model state...")
    try:
        initial_state_dict = torch.load('/tmp/bshd_initial_model_state.pt', map_location='cpu')
        model.load_state_dict(initial_state_dict)
        print(f"Rank {dist_config.rank}: Model state loaded successfully")
    except Exception as e:
        print(f"Rank {dist_config.rank}: Error loading model state: {e}")
        raise

    model = model.to(device)
    print(f"Rank {dist_config.rank}: Model moved to device")
    # TODO: Set context parallel group on the model. 

    print("Setting up DDP...")
    group_fsdp_cp = device_mesh[("fsdp", "cp")]._flatten("dp_cp").get_group()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[dist_config.local_rank],
        output_device=dist_config.local_rank,
        process_group=group_fsdp_cp,
    )
    print(f"Rank {dist_config.rank}: DDP setup complete")

    cp_group = device_mesh["cp"].get_group()
    print(f"Rank {dist_config.rank}: CP group obtained")

    # Set context parallel group on the model.
    if cp_size > 1:
        print("Setting up context parallel groups...")
        for i, transformer_layer in enumerate(model.module.transformer_layers):
            print(f"Rank {dist_config.rank}: Setting CP group for layer {i}")
            transformer_layer.set_context_parallel_group(
                # Context parallel group, and GLOBAL ranks participating in this group.
                cp_group,
                torch.distributed.get_process_group_ranks(device_mesh["cp"].get_group()),
                torch.cuda.Stream()
            )
        print(f"Rank {dist_config.rank}: All CP groups set up")

    # Synchronize all ranks before proceeding
    print("Synchronizing ranks before forward pass...")
    dist.barrier()
    print(f"Rank {dist_config.rank}: Barrier passed, ready for forward pass")

    # Run a forward pass with deterministic seeding
    torch.manual_seed(42)  # Set seed for forward pass randomness (dropout, etc.)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    batch = get_dummy_data_bshd()
    
    import sys; sys.exit(0)
    # TODO: Try the version of this that uses BSHD instead of the THD version.
    input_ids_padded, labels_padded, position_ids_padded = get_thd_batch_on_this_cp_rank(
            batch['cu_seqlens_q_padded'], batch['cu_seqlens_kv_padded'], batch['input_ids_padded'], batch['labels_padded'], batch['position_ids_padded'], 
            cp_group
        )

    batch['input_ids'] = input_ids_padded
    batch['labels'] = labels_padded
    batch['position_ids'] = position_ids_padded
    
    batch = {k: v.to(device, non_blocking=True).contiguous() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # Debug: Print final batch info going into model
    print(f"Rank {dist_config.rank}: Final batch going into model:")
    print(f"  input_ids shape: {batch['input_ids'].shape if 'input_ids' in batch else 'N/A'}")
    print(f"  labels shape: {batch['labels'].shape if 'labels' in batch else 'N/A'}")
    print(f"  cu_seqlens_q: {batch['cu_seqlens_q']}")
    print(f"  cu_seqlens_q_padded: {batch['cu_seqlens_q_padded']}")

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(batch)

    print("output of CP2 rank shape is :", f"{dist_config.rank}-{output.shape}")
    print(f"Rank {dist_config.rank}: CP=2 output shape: {output.shape}")
    logits_flat = output.view(-1, config.vocab_size)
    labels_flat = batch["labels"].view(-1)
    valid_mask = labels_flat != -100
    if valid_mask.any():
        loss = F.cross_entropy(logits_flat[valid_mask], labels_flat[valid_mask], reduction="mean")
    else:
        loss = logits_flat.sum() * 0.0

    print(f"Rank {dist_config.rank}: CP=2 loss: {loss.item():.6f}")

    loss.backward()

    grads = collect_gradients(model, layer_patterns=target_layers, max_params=15)
    print(f"Rank {dist_config.rank}: CP=2 collected {len(grads)} parameter gradients")

    # Store CP=2 results (only what we need to test)
    cp2_results = {
        'logits': output.clone().detach().cpu(),
        'loss': loss.clone().detach().cpu(),
        'grad_norms': {name: grad.norm().item() for name, grad in grads.items()},
        'grads': grads,
    }

    print(f"Rank {dist_config.rank}: CP=2 results: logits {cp2_results['logits'].shape}, loss {cp2_results['loss'].item():.6f}, {len(cp2_results['grad_norms'])} grad norms")

    # I want to save the CP2 rank0 and CP2 rank 1 results to a file.
    torch.save(cp2_results, f'/tmp/bshd_cp2_rank_{dist_config.rank}_results.pt')
    dist.barrier()
    print("Done saving CP=2 results")
    # Clean up CP=2
    dist.destroy_process_group()
