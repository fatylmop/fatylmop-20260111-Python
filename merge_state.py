import torch
import flashinfer

def save_merge_state_golden():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define data dimensions
    seq_len = 2048
    num_heads = 32
    head_dim = 128

    # Create input tensors
    v_a = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")
    s_a = torch.randn(seq_len, num_heads, dtype=torch.float32).to("cuda:0")
    v_b = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")
    s_b = torch.randn(seq_len, num_heads, dtype=torch.float32).to("cuda:0")

    # Run merge_state
    with torch.no_grad():
        v_merged, s_merged = flashinfer.cascade.merge_state(v_a, s_a, v_b, s_b)

    print(f"v_merged shape: {v_merged.shape}")
    print(f"s_merged shape: {s_merged.shape}")
    # Save golden data
    torch.save({
        "v_a": v_a,
        "s_a": s_a,
        "v_b": v_b,
        "s_b": s_b,
        "v_merged": v_merged,
        "s_merged": s_merged
    }, "/home/cat3060/FlashinferTest/golden/merge_state_golden.pth")

    print("The golden result has been saved!")

def test_merge_state():
    # Load golden data
    data = torch.load("/home/cat3060/FlashinferTest/golden/merge_state_golden.pth")
    v_a = data["v_a"].cuda()
    s_a = data["s_a"].cuda()
    v_b = data["v_b"].cuda()
    s_b = data["s_b"].cuda()
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
                _ = flashinfer.cascade.merge_state(v_a, s_a, v_b, s_b)
        except:
            pass

    # Test loop
    for run in range(num_runs):
        try:
            with torch.no_grad():
                v_merged, s_merged = flashinfer.cascade.merge_state(v_a, s_a, v_b, s_b)
                
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
    # save_merge_state_golden()
    test_merge_state()


# 806mV
'''
Total runs: 1000
Total errors: 1
- Correct runs: 999 (99.900%)
- CUDA Errors: 0 (0.000%)
- Output Mismatches: 1 (0.100%)
- Other Errors: 0 (0.000%)
'''


# 800mV
'''
Total runs: 1000
Total errors: 12
- Correct runs: 988 (98.800%)
- CUDA Errors: 0 (0.000%)
- Output Mismatches: 12 (1.200%)
- Other Errors: 0 (0.000%)

Total runs: 1000
Total errors: 24
- Correct runs: 976 (97.600%)
- CUDA Errors: 0 (0.000%)
- Output Mismatches: 24 (2.400%)
- Other Errors: 0 (0.000%)

Total runs: 1000
Total errors: 22
- Correct runs: 978 (97.800%)
- CUDA Errors: 0 (0.000%)
- Output Mismatches: 22 (2.200%)
- Other Errors: 0 (0.000%)

Total runs: 1000
Total errors: 12
- Correct runs: 988 (98.800%)
- CUDA Errors: 0 (0.000%)
- Output Mismatches: 12 (1.200%)
- Other Errors: 0 (0.000%)

Total runs: 1000
Total errors: 22
- Correct runs: 978 (97.800%)
- CUDA Errors: 0 (0.000%)
- Output Mismatches: 22 (2.200%)
- Other Errors: 0 (0.000%)
'''


# 793mV
'''
Total runs: 1000
Total errors: 828
- Correct runs: 172 (17.200%)
- CUDA Errors: 1 (0.100%)
- Output Mismatches: 22 (2.200%)
- Other Errors: 805 (80.500%)

Total runs: 1000
Total errors: 784
- Correct runs: 216 (21.600%)
- CUDA Errors: 1 (0.100%)
- Output Mismatches: 21 (2.100%)
- Other Errors: 762 (76.200%)

Total runs: 1000
Total errors: 885
- Correct runs: 115 (11.500%)
- CUDA Errors: 1 (0.100%)
- Output Mismatches: 4 (0.400%)
- Other Errors: 880 (88.000%)

Total runs: 1000
Total errors: 860
- Correct runs: 140 (14.000%)
- CUDA Errors: 1 (0.100%)
- Output Mismatches: 10 (1.000%)
- Other Errors: 849 (84.900%)

Total runs: 1000
Total errors: 890
- Correct runs: 110 (11.000%)
- CUDA Errors: 1 (0.100%)
- Output Mismatches: 9 (0.900%)
- Other Errors: 880 (88.000%)
'''