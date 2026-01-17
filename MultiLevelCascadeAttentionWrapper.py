import torch
import flashinfer

def save_cascade_attention_golden():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define data dimensions
    num_levels = 2
    batch_size = 4
    head_nums = 32
    head_dim = 128
    num_layers = 1  # Test single layer
    page_size = 16
    shared_kv_num_pages = 512
    unique_kv_num_pages = 128
    total_num_pages = shared_kv_num_pages + unique_kv_num_pages

    # Allocate workspace buffer (128MB)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")

    # Initialize cascade attention wrapper
    wrapper = flashinfer.cascade.MultiLevelCascadeAttentionWrapper(
        num_levels, workspace_buffer, "NHD"
    )

    # Create query and page table data for two levels
    qo_indptr_arr = [
        torch.tensor([0, batch_size], dtype=torch.int32, device="cuda:0"),  # Shared level
        torch.arange(batch_size + 1, dtype=torch.int32, device="cuda:0")   # Unique level
    ]
    paged_kv_indptr_arr = [
        torch.tensor([0, shared_kv_num_pages], dtype=torch.int32, device="cuda:0"),  # Shared level
        torch.tensor([0, 32, 64, 96, 128], dtype=torch.int32, device="cuda:0")      # Unique level
    ]
    paged_kv_indices_arr = [
        torch.arange(shared_kv_num_pages).int().to("cuda:0"),  # Shared level
        torch.arange(shared_kv_num_pages, total_num_pages).int().to("cuda:0")  # Unique level
    ]
    paged_kv_last_page_len_arr = [
        torch.tensor([page_size], dtype=torch.int32, device="cuda:0"),  # Shared level
        torch.tensor([16, 16, 16, 16], dtype=torch.int32, device="cuda:0")  # Unique level
    ]

    # Create query and key-value cache for single layer
    q = torch.randn(batch_size, head_nums, head_dim).half().to("cuda:0")
    kv_cache = [
        torch.randn(
            total_num_pages, 2, page_size, head_nums, head_dim,
            dtype=torch.float16, device="cuda:0"
        ) for _ in range(num_layers)
    ]

    # Plan the cascade attention operation
    wrapper.plan(
        qo_indptr_arr,
        paged_kv_indptr_arr,
        paged_kv_indices_arr,
        paged_kv_last_page_len_arr,
        head_nums,  # num_qo_heads
        head_nums,  # num_kv_heads
        head_dim,
        page_size,
        causal=True,
        pos_encoding_mode="NONE",
        q_data_type=torch.float16
    )

    # Run cascade attention for single layer
    with torch.no_grad():
        output_golden = wrapper.run(q, kv_cache[0])

    print(output_golden.shape)
    # Save golden data
    torch.save({
        "q": q,
        "kv_cache": kv_cache[0],  # Save single layer cache
        "qo_indptr_arr": qo_indptr_arr,
        "paged_kv_indptr_arr": paged_kv_indptr_arr,
        "paged_kv_indices_arr": paged_kv_indices_arr,
        "paged_kv_last_page_len_arr": paged_kv_last_page_len_arr,
        "output_golden": output_golden  # Save single layer output
    }, "/home/cat3060/FlashinferTest/golden/multi_level_cascade_attention_golden.pth")

    print("The golden result has been saved!")

def test_cascade_attention():
    # Load golden data
    data = torch.load("/home/cat3060/FlashinferTest/golden/multi_level_cascade_attention_golden.pth")
    q = data["q"].cuda()
    kv_cache = data["kv_cache"].cuda()
    qo_indptr_arr = [x.cuda() for x in data["qo_indptr_arr"]]
    paged_kv_indptr_arr = [x.cuda() for x in data["paged_kv_indptr_arr"]]
    paged_kv_indices_arr = [x.cuda() for x in data["paged_kv_indices_arr"]]
    paged_kv_last_page_len_arr = [x.cuda() for x in data["paged_kv_last_page_len_arr"]]
    output_golden = data["output_golden"].cuda()

    # Define data dimensions
    num_levels = 2
    batch_size = 4
    head_nums = 32
    head_dim = 128
    num_layers = 1
    page_size = 16

    # Allocate workspace buffer (128MB)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")

    # Initialize cascade attention wrapper
    wrapper = flashinfer.cascade.MultiLevelCascadeAttentionWrapper(
        num_levels, workspace_buffer, "NHD"
    )

    # Plan the cascade attention operation
    wrapper.plan(
        qo_indptr_arr,
        paged_kv_indptr_arr,
        paged_kv_indices_arr,
        paged_kv_last_page_len_arr,
        head_nums,  # num_qo_heads
        head_nums,  # num_kv_heads
        head_dim,
        page_size,
        causal=True,
        pos_encoding_mode="NONE",
        q_data_type=torch.float16
    )

    # Initialize counters
    num_runs = 1000
    cuda_errors = 0
    normal_errors = 0
    mismatch_errors = 0

    print(f"Golden output shape: {output_golden.shape}")

    # Warm up GPU
    print("Warming up GPU...")
    for _ in range(50):
        try:
            with torch.no_grad():
                _ = wrapper.run(q, kv_cache)
        except:
            pass

    # Test loop
    for run in range(num_runs):
        try:
            with torch.no_grad():
                output = wrapper.run(q, kv_cache)
                
            # Check result
            match = torch.allclose(output, output_golden, rtol=1e-3, atol=1e-3)
            print(f"Run {run + 1}, Match: {match}")
            if not match:
                mismatch_errors += 1
                print(f"Run {run + 1}: Output mismatch")

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
    # save_cascade_attention_golden()
    test_cascade_attention()

# 800mV
'''
大多都是100% crash
'''


# 806mV
'''
Total runs: 1000
Total errors: 2
- Correct runs: 998 (99.800%)
- CUDA Errors: 0 (0.000%)
- Output Mismatches: 2 (0.200%)
- Other Errors: 0 (0.000%)

Total runs: 1000
Total errors: 1
- Correct runs: 999 (99.900%)
- CUDA Errors: 0 (0.000%)
- Output Mismatches: 1 (0.100%)
- Other Errors: 0 (0.000%)
'''