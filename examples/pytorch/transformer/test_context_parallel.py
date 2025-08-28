# The actual testing file
import torch

def calculate_cp_indices(cu_seqlens_padded, cp_size=2):
    """
    Calculate which token indices each CP rank should process.
    This matches the ORIGINAL logic from lightweight_test.py and integration_test_cp_dp.py
    
    Args:
        cu_seqlens_padded: Cumulative sequence lengths tensor
        cp_size: Context parallel size (default 2)
    
    Returns:
        dict: {rank_id: [list of token indices]}
    """
    total_slices_per_sequence = 2 * cp_size  # 4 slices per sequence for CP=2
    num_sequences = len(cu_seqlens_padded) - 1
    
    rank_indices = {i: [] for i in range(cp_size)}
    
    for i in range(num_sequences):
        seq_start = cu_seqlens_padded[i].item()
        seq_end = cu_seqlens_padded[i+1].item()
        seq_len = seq_end - seq_start
        slice_size = seq_len // total_slices_per_sequence
        
        # Rank 0 gets: slice 0 + slice 3 (first + last slices of each sequence)
        rank_indices[0].extend(range(seq_start + (0 * slice_size), seq_start + (1 * slice_size)))
        rank_indices[0].extend(range(seq_start + (3 * slice_size), seq_start + (4 * slice_size)))
        
        # Rank 1 gets: slice 1 + slice 2 (middle slices)  
        rank_indices[1].extend(range(seq_start + (1 * slice_size), seq_start + (2 * slice_size)))
        rank_indices[1].extend(range(seq_start + (2 * slice_size), seq_start + (3 * slice_size)))
    
    return rank_indices

def reconstruct_cp_logits(cp2_rank0_logits, cp2_rank1_logits, cp1_baseline_logits, cu_seqlens_padded, cp_size=2):
    """
    Reconstruct full sequence logits from distributed CP results.
    
    Args:
        cp2_rank0_logits: Logits tensor from CP rank 0 [num_tokens_rank0, vocab_size]
        cp2_rank1_logits: Logits tensor from CP rank 1 [num_tokens_rank1, vocab_size]
        cp1_baseline_logits: Full sequence logits from CP=1 run [total_tokens, vocab_size]
        cu_seqlens_padded: Cumulative sequence lengths tensor
        cp_size: Context parallel size (default 2)
    
    Returns:
        torch.Tensor: Reconstructed full sequence logits [total_tokens, vocab_size]
    """
    print(f"🔄 Reconstructing CP={cp_size} logits from 2 ranks")
    print(f"  Input shapes: rank0={cp2_rank0_logits.shape}, rank1={cp2_rank1_logits.shape}, baseline={cp1_baseline_logits.shape}")
    
    # Calculate expected indices for each rank
    rank_indices = calculate_cp_indices(cu_seqlens_padded, cp_size)
    
    rank0_indices = rank_indices[0]
    rank1_indices = rank_indices[1]
    
    print(f"  Expected tokens - rank0: {len(rank0_indices)}, rank1: {len(rank1_indices)}")
    print(f"  Actual tokens - rank0: {cp2_rank0_logits.shape[0]}, rank1: {cp2_rank1_logits.shape[0]}")
    
    # Verify sizes match expectations
    if len(rank0_indices) != cp2_rank0_logits.shape[0]:
        print(f"    ❌ Rank 0 size mismatch: expected {len(rank0_indices)}, got {cp2_rank0_logits.shape[0]}")
        return None
        
    if len(rank1_indices) != cp2_rank1_logits.shape[0]:
        print(f"    ❌ Rank 1 size mismatch: expected {len(rank1_indices)}, got {cp2_rank1_logits.shape[0]}")
        return None
    
    # Reconstruct full logits
    reconstructed_logits = torch.zeros_like(cp1_baseline_logits)
    
    # Place rank 0 logits
    reconstructed_logits[rank0_indices] = cp2_rank0_logits
    print(f"    ✅ Rank 0: placed {len(rank0_indices)} tokens successfully")
    
    # Place rank 1 logits  
    reconstructed_logits[rank1_indices] = cp2_rank1_logits
    print(f"    ✅ Rank 1: placed {len(rank1_indices)} tokens successfully")
    
    print(f"  Reconstructed logits shape: {reconstructed_logits.shape}")
    return reconstructed_logits

def compare_logits(reconstructed_logits, baseline_logits, name="Reconstructed"):
    """
    Compare two sets of logits and print detailed statistics.
    
    Args:
        reconstructed_logits: Reconstructed logits tensor
        baseline_logits: Baseline logits tensor  
        name: Name for the comparison (for printing)
    """
    logits_abs_diff = torch.abs(reconstructed_logits - baseline_logits)
    logits_diff = logits_abs_diff.max().item()
    logits_mean_diff = logits_abs_diff.mean().item()
    
    total_elements = reconstructed_logits.numel()
    within_1e3 = (logits_abs_diff < 1e-3).sum().item()
    within_1e2 = (logits_abs_diff < 1e-2).sum().item()
    within_2e2 = (logits_abs_diff < 2e-2).sum().item()
    within_5e2 = (logits_abs_diff < 5e-2).sum().item()
    
    print(f"\n📊 {name} Logits Comparison:")
    print(f"  Total elements: {total_elements}")
    print(f"  Max difference: {logits_diff:.8f}")
    print(f"  Mean difference: {logits_mean_diff:.8f}")
    print(f"  Elements within 1e-3: {within_1e3}/{total_elements} ({100*within_1e3/total_elements:.1f}%)")
    print(f"  Elements within 1e-2: {within_1e2}/{total_elements} ({100*within_1e2/total_elements:.1f}%)")
    print(f"  Elements within 2e-2: {within_2e2}/{total_elements} ({100*within_2e2/total_elements:.1f}%)")
    print(f"  Elements within 5e-2: {within_5e2}/{total_elements} ({100*within_5e2/total_elements:.1f}%)")
    
    return {
        'max_diff': logits_diff,
        'mean_diff': logits_mean_diff,
        'within_2e2_pct': 100 * within_2e2 / total_elements
    }

# Load saved results
cp2_rank0_results = torch.load('/tmp/cp2_rank_0_results.pt')
cp2_rank1_results = torch.load('/tmp/cp2_rank_1_results.pt')
cp1_results = torch.load('/tmp/cp1_results.pt')
data = torch.load('/tmp/data.pt')

print("Keys of all data loaded")
print(data.keys())
print("Keys of cp2_rank0_results")
print(cp2_rank0_results.keys())
print("Keys of cp2_rank1_results")
print(cp2_rank1_results.keys())
print("Keys of cp1_results")
print(cp1_results.keys())

print(f"CP=1 logits shape: {cp1_results['logits'].shape}")
print(f"CP=2 rank 0 logits shape: {cp2_rank0_results['logits'].shape}")
print(f"CP=2 rank 1 logits shape: {cp2_rank1_results['logits'].shape}")

# Organize CP=2 results by rank
cp2_results_by_rank = {
    0: cp2_rank0_results,
    1: cp2_rank1_results
}

cp2_rank0_logits = cp2_rank0_results['logits']
cp2_rank1_logits = cp2_rank1_results['logits']
cp1_logits = cp1_results['logits']

print("data types of logits")
print(type(cp2_rank0_logits))
print(type(cp2_rank1_logits))
print(type(cp1_logits))

print("Shapes of logits")
print(cp2_rank0_logits.shape)
print(cp2_rank1_logits.shape)
print(cp1_logits.shape)

# Reconstruct CP=2 logits from both ranks
cu_seqlens_padded = data['cu_seqlens_q_padded']
print(f"cu_seqlens_padded: {cu_seqlens_padded}")

reconstructed_cp2_logits = reconstruct_cp_logits(
    cp2_rank0_logits,
    cp2_rank1_logits, 
    cp1_logits, 
    cu_seqlens_padded
)


print("reconstructed_cp2_logits", reconstructed_cp2_logits.shape)
print("cp1_results['logits']", cp1_results['logits'].shape)

if reconstructed_cp2_logits is not None:
    # Compare reconstructed CP=2 with CP=1 baseline
    comparison_stats = compare_logits(reconstructed_cp2_logits, cp1_results['logits'], "CP=2 vs CP=1")
    
    # Test passes if >95% of elements are within 2e-2 tolerance
    if comparison_stats['within_2e2_pct'] >= 95.0:
        print(f"\n🎉 LOGITS TEST PASSED: {comparison_stats['within_2e2_pct']:.1f}% within tolerance")
    else:
        print(f"\n❌ LOGITS TEST FAILED: Only {comparison_stats['within_2e2_pct']:.1f}% within tolerance (need ≥95%)")
else:
    print(f"\n❌ RECONSTRUCTION FAILED: Cannot compare logits due to size mismatches")








