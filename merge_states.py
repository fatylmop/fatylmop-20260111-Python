import torch
import flashinfer

def save_merge_states_golden():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define data dimensions
    seq_len = 2048
    num_states = 100
    num_heads = 32
    head_dim = 128

    # Create input tensors
    v = torch.randn(seq_len, num_states, num_heads, head_dim).half().to("cuda:0")
    s = torch.randn(seq_len, num_states, num_heads, dtype=torch.float32).to("cuda:0")

    # Run merge_states
    with torch.no_grad():
        v_merged, s_merged = flashinfer.cascade.merge_states(v, s)

    print(f"v_merged shape: {v_merged.shape}")
    print(f"s_merged shape: {s_merged.shape}")
    # Save golden data
    torch.save({
        "v": v,
        "s": s,
        "v_merged": v_merged,
        "s_merged": s_merged
    }, "/home/cat3060/FlashinferTest/golden/merge_states_golden.pth")

    print("The golden result has been saved!")

def test_merge_states():
    # Load golden data
    data = torch.load("/home/cat3060/FlashinferTest/golden/merge_states_golden.pth")
    v = data["v"].cuda()
    s = data["s"].cuda()
    v_merged_golden = data["v_merged"].cuda()
    s_merged_golden = data["s_merged"].cuda()

    # Initialize counters
    num_runs = 1000
    cuda_errors = 0
    normal_errors = 0
    mismatch_errors = 0

    print(f"Golden v_merged shape: {v_merged_golden.shape}")
    print(f"Golden s_merged shape: {s_merged_golden.shape}")

    # Warm up GPU
    print("Warming up GPU...")
    for _ in range(50):
        try:
            with torch.no_grad():
                _ = flashinfer.cascade.merge_states(v, s)
        except:
            pass

    # Test loop
    for run in range(num_runs):
        try:
            with torch.no_grad():
                v_merged, s_merged = flashinfer.cascade.merge_states(v, s)
                
            # Check results for both v_merged and s_merged
            v_match = torch.allclose(v_merged, v_merged_golden, rtol=1e-3, atol=1e-3)
            s_match = torch.allclose(s_merged, s_merged_golden, rtol=1e-3, atol=1e-3)
            match = v_match and s_match
            print(f"Run {run + 1}, Match: {match}")
            if not match:
                mismatch_errors += 1
                print(f"Run {run + 1}: Output mismatch (v_match: {v_match}, s_match: {s_match})")

        except RuntimeError as e:
            if 'CUDA' in str(e) or 'cudaError' in str(e):
                cuda_errors += 1
            else:
                normal_errors += 1

    # Print statistics
    total_errors = cuda_errors + normal_errors + mismatch_errors
    print(f"\nTotal runs: {num_runs}")
    print(f"Total errors: {total_errors}")
    print(f"- Correct runs: {num_runs - total_errors} ({(num_runs - total_errors)/num_runs*100:.3f}%)")
    print(f"- CUDA Errors: {cuda_errors} ({cuda_errors/num_runs*100:.3f}%)")
    print(f"- Output Mismatches: {mismatch_errors} ({mismatch_errors/num_runs*100:.3f}%)")
    print(f"- Other Errors: {normal_errors} ({normal_errors/num_runs*100:.3f}%)")

if __name__ == "__main__":
    # Uncomment the function you want to run
    # save_merge_states_golden()
    test_merge_states()

# 806mV
'''
Total runs: 1000
Total errors: 475
- Correct runs: 525 (52.500%)
- CUDA Errors: 1 (0.100%)
- Output Mismatches: 2 (0.200%)
- Other Errors: 472 (47.200%)

Total runs: 1000
Total errors: 532
- Correct runs: 468 (46.800%)
- CUDA Errors: 1 (0.100%)
- Output Mismatches: 1 (0.100%)
- Other Errors: 530 (53.000%)

Total runs: 1000
Total errors: 597
- Correct runs: 403 (40.300%)
- CUDA Errors: 1 (0.100%)
- Output Mismatches: 0 (0.000%)
- Other Errors: 596 (59.600%)
'''

