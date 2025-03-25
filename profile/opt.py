import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np

# Load model
model_name = "facebook/opt-6.7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)

model.eval()

seq_lengths = [32, 64, 128, 256, 512]
results = []

def count_llm_flops(seq_len, vocab_size, hidden_size, num_layers):
    # Simplified FLOP estimation for decoder-only transformer
    # Forward pass through embedding + position
    embedding_flops = seq_len * hidden_size

    # Each transformer layer
    per_layer_flops = (
        # Self-attention: 4 * seq_len * hidden_size^2 for Q,K,V,O projections
        4 * seq_len * hidden_size * hidden_size +
        # Self-attention: seq_len^2 * hidden_size for attention matrix multiply
        seq_len * seq_len * hidden_size +
        # FFN: 8 * seq_len * hidden_size^2 (assuming 4x expansion in FFN)
        8 * seq_len * hidden_size * hidden_size
    )

    # Total for all layers
    total_layer_flops = num_layers * per_layer_flops

    # Final LM head projection
    lm_head_flops = seq_len * hidden_size * vocab_size

    return embedding_flops + total_layer_flops + lm_head_flops

# For OPT-6.7B
hidden_size = 4096
num_layers = 32
vocab_size = 50272  # OPT vocabulary size from Huggingface

# Track memory transactions
def measure_memory_bandwidth(prompt, max_new_tokens=1):
    torch.cuda.reset_peak_memory_stats(0)  # Assuming device 0
    torch.cuda.empty_cache()

    input_tokens = tokenizer(prompt, return_tensors="pt").to('cuda')
    seq_len = input_tokens.input_ids.shape[1] + max_new_tokens

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for _ in range(3):
            _ = model.generate(**input_tokens, max_new_tokens=max_new_tokens)

    start.record()
    with torch.no_grad():
        output = model.generate(**input_tokens, max_new_tokens=max_new_tokens)
    end.record()

    torch.cuda.synchronize()
    time_ms = start.elapsed_time(end)

    # Estimate memory access
    # This is a very rough approximation based on model size and sequence processing
    total_params_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    input_size = input_tokens.input_ids.nelement() * input_tokens.input_ids.element_size()

    # LLM KV cache overhead - approximately 2 * num_layers * seq_len * hidden_size * 2 bytes (fp16)
    kv_cache_size = 2 * num_layers * seq_len * hidden_size * 2

    # LLMs typically reuse parameters but access KV cache extensively
    # Only a fraction of parameters need to be loaded for each token
    param_reuse_factor = 0.4  # Approximate
    bytes_accessed = input_size + (total_params_bytes * param_reuse_factor) + kv_cache_size

    flops = count_llm_flops(seq_len, vocab_size, hidden_size, num_layers)

    bandwidth = bytes_accessed / (time_ms / 1000)

    return {
        'seq_len': seq_len,
        'time_ms': time_ms,
        'flops': flops,
        'bytes_accessed': bytes_accessed,
        'bandwidth': bandwidth,
        'flops_per_byte': flops / bytes_accessed,
        'gflops_per_second': flops / (time_ms / 1000) / 1e9
    }

print("Running roofline analysis for OPT-6.7B...")
base_prompt = "Columbia University is the best "

for prompt_len in seq_lengths:
    if prompt_len <= len(base_prompt.split()):
        words = base_prompt.split()[:prompt_len]
        prompt = ' '.join(words)
    else:
        repetitions = prompt_len // len(base_prompt.split()) + 1
        words = (base_prompt.split() * repetitions)[:prompt_len]
        prompt = ' '.join(words)

    result = measure_memory_bandwidth(prompt)
    results.append(result)

    print(f"Sequence length: {result['seq_len']}, "
          f"GFLOPS/s: {result['gflops_per_second']:.2f}, "
          f"Operational Intensity: {result['flops_per_byte']:.2f} FLOPS/Byte")

plt.figure(figsize=(12, 8))

'''
GPU Specs
A100
peak_fp32_performance = 19.5  # TFLOPS
peak_fp16_performance = 312  # TFLOPS
memory_bandwidth = 1.6  # TB/s (1600 GB/s)

T4
peak_fp32_performance = 8.1  # TFLOPS
peak_fp16_performance = 65  # TFLOPS
memory_bandwidth = 0.3  # TB/s (300 GB/s)

L4
peak_fp32_performance = 30.3  # TFLOPS
peak_fp16_performance = 242  # TFLOPS
memory_bandwidth = 0.3  # TB/s (1600 GB/s)
'''
peak_fp32_performance = 8.1  # TFLOPS
peak_fp16_performance = 65  # TFLOPS
memory_bandwidth = 0.3  # TB/s (300 GB/s)

peak_fp32_gflops = peak_fp32_performance * 1000  # Convert TFLOPS to GFLOPS
peak_fp16_gflops = peak_fp16_performance * 1000  # Convert TFLOPS to GFLOPS
memory_bw_gb = memory_bandwidth * 1000  # Convert TB/s to GB/s

# Create the roofline
x_range = np.logspace(-1, 3, 1000)  # Operational intensity range (FLOPS/Byte)

memory_bound = [x * memory_bw_gb for x in x_range]

compute_bound_fp32 = [peak_fp32_gflops for _ in x_range]
compute_bound_fp16 = [peak_fp16_gflops for _ in x_range]

ridge_point_fp32 = peak_fp32_gflops / memory_bw_gb
ridge_point_fp16 = peak_fp16_gflops / memory_bw_gb

# Plot the roofline model
plt.loglog(x_range, memory_bound, 'b-', label='Memory Bandwidth Limit')
plt.loglog(x_range, compute_bound_fp32, 'r-', label='FP32 Compute Limit')
plt.loglog(x_range, compute_bound_fp16, 'g-', label='FP16 Compute Limit')

for r in results:
    plt.loglog(r['flops_per_byte'], r['gflops_per_second'], 'ko', markersize=10)
    plt.annotate(f"Seq={r['seq_len']}",
                 (r['flops_per_byte'], r['gflops_per_second']),
                 xytext=(5, 5), textcoords='offset points')

plt.axvline(x=ridge_point_fp32, color='r', linestyle='--', alpha=0.3)
plt.axvline(x=ridge_point_fp16, color='g', linestyle='--', alpha=0.3)

plt.xlabel('FLOPS/Byte')
plt.ylabel('GFLOPS/s')
plt.title('Roofline Model for OPT-6.7B on NVIDIA T4')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()

# Add annotations for the A100 specs
plt.annotate(f"FP32 Peak: {peak_fp32_performance} TFLOPS",
             (ridge_point_fp32*5, peak_fp32_gflops),
             xytext=(0, 10), textcoords='offset points')
plt.annotate(f"FP16 Peak: {peak_fp16_performance} TFLOPS",
             (ridge_point_fp16*5, peak_fp16_gflops),
             xytext=(0, 10), textcoords='offset points')
plt.annotate(f"Memory BW: {memory_bandwidth} TB/s",
             (0.5, memory_bw_gb*0.5),
             xytext=(10, 0), textcoords='offset points')

plt.savefig('opt_roofline.png')
plt.show()

print("Roofline analysis complete. Check 'opt_roofline.png'")