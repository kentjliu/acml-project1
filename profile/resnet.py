import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False).to(device)
model.eval()

batch_sizes = [1, 8, 16, 32, 64]
results = []

def count_resnet50_flops(batch_size):
    # ResNet50 has approximately 4 billion FLOPs per image at 224x224
    return batch_size * 4e9

def measure_memory_bandwidth(func, batch_size, input_tensor):
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    mem_before = torch.cuda.memory_allocated(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    result = func(input_tensor)
    end.record()

    torch.cuda.synchronize()
    time_ms = start.elapsed_time(end)

    mem_after = torch.cuda.max_memory_allocated(device)

    # Estimate memory traffic (this is an approximation)
    # Input memory: input size + weights loaded
    # Output memory: output size
    input_size = input_tensor.nelement() * input_tensor.element_size()
    output_size = result.nelement() * result.element_size()
    total_params = sum(p.nelement() * p.element_size() for p in model.parameters())

    # Estimate total bytes accessed (this is a rough approximation)
    bytes_accessed = input_size + output_size + total_params

    # Memory bandwidth in bytes/second
    bandwidth = bytes_accessed / (time_ms / 1000)

    return {
        'time_ms': time_ms,
        'flops': count_resnet50_flops(batch_size),
        'bytes_accessed': bytes_accessed,
        'bandwidth': bandwidth,
        'flops_per_byte': count_resnet50_flops(batch_size) / bytes_accessed,
        'gflops_per_second': count_resnet50_flops(batch_size) / (time_ms / 1000) / 1e9
    }

print("Running roofline analysis...")
for batch_size in batch_sizes:
    input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_tensor)

    # Actual measurement
    with torch.no_grad():
        result = measure_memory_bandwidth(model, batch_size, input_tensor)
        results.append({
            'batch_size': batch_size,
            **result
        })
        print(f"Batch size: {batch_size}, "
              f"GFLOPS/s: {result['gflops_per_second']:.2f}, "
              f"Operational Intensity: {result['flops_per_byte']:.2f} FLOPS/Byte")

# Plot roofline model
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

# Convert to consistent units
peak_fp32_gflops = peak_fp32_performance * 1000  
peak_fp16_gflops = peak_fp16_performance * 1000 
memory_bw_gb = memory_bandwidth * 1000

# Create the roofline
x_range = np.logspace(-1, 3, 1000)  

memory_bound = [x * memory_bw_gb for x in x_range]

compute_bound_fp32 = [peak_fp32_gflops for _ in x_range]
compute_bound_fp16 = [peak_fp16_gflops for _ in x_range]

# Ridge points (where memory bound meets compute bound)
ridge_point_fp32 = peak_fp32_gflops / memory_bw_gb
ridge_point_fp16 = peak_fp16_gflops / memory_bw_gb

# Plot the roofline model
plt.loglog(x_range, memory_bound, 'b-', label='Memory Bandwidth Limit')
plt.loglog(x_range, compute_bound_fp32, 'r-', label='FP32 Compute Limit')
plt.loglog(x_range, compute_bound_fp16, 'g-', label='FP16 Compute Limit')

for r in results:
    plt.loglog(r['flops_per_byte'], r['gflops_per_second'], 'ko', markersize=10)
    plt.annotate(f"BS={r['batch_size']}",
                 (r['flops_per_byte'], r['gflops_per_second']),
                 xytext=(5, 5), textcoords='offset points')

plt.axvline(x=ridge_point_fp32, color='r', linestyle='--', alpha=0.3)
plt.axvline(x=ridge_point_fp16, color='g', linestyle='--', alpha=0.3)

plt.xlabel('FLOPS/Byte')
plt.ylabel('GFLOPS/s')
plt.title('Roofline Model for ResNet50 on NVIDIA T4')
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

plt.savefig('resnet50_roofline.png')
plt.show()

print("Roofline analysis complete. Check 'resnet50_roofline.png'")
