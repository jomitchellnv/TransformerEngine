from utils import get_dummy_data, collect_gradients
from model import SimpleConfig, SimpleModel
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import get_thd_batch_on_this_cp_rank
import random
import numpy as np
import torch
import os
import torch.nn.functional as F

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

data = get_dummy_data()


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
model_cp1 = SimpleModel(config)
model_cp1 = model_cp1.to(device)

# No context parallel setup needed since CP=1
# No DDP wrapper needed since we're running on single GPU

# Run the same test with CP=1 using same random seed
torch.manual_seed(42)  # Same seed as CP=2 for identical randomness
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

batch = get_dummy_data()
batch['input_ids'] = batch['input_ids_padded']
batch['labels'] = batch['labels_padded']
batch['position_ids'] = batch['position_ids_padded']
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
torch.save(initial_state_dict, '/tmp/initial_model_state.pt')
print("Model state saved for CP=2 reuse")

# Store CP=1 results (only what we need to test)
cp1_results = {
    'logits': output_cp1.clone().detach().cpu(),
    'loss': loss_cp1.clone().detach().cpu(),
    'grad_norms': {name: grad.norm().item() for name, grad in grads_cp1.items()},
    'grads': grads_cp1,
}

print(f"CP=1 results: logits {cp1_results['logits'].shape}, loss {cp1_results['loss'].item():.6f}, {len(cp1_results['grad_norms'])} grad norms")

# No process group to destroy for CP=1 (single GPU run)
print("CP=1 baseline completed successfully")

