import matplotlib.pyplot as plt
import numpy as np

# GPU specs
'''
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
peak_fp32_performance = 30.3  # TFLOPS
peak_fp16_performance = 242  # TFLOPS
memory_bandwidth = 0.3  # TB/s (1600 GB/s)

peak_fp32_gflops = peak_fp32_performance * 1000 
peak_fp16_gflops = peak_fp16_performance * 1000 
memory_bw_gb = memory_bandwidth * 1000 

# Data
opt_data = [
    {'seq_len': 35, 'gflops': 3659.02, 'intensity': 43.56},
    {'seq_len': 67, 'gflops': 6601.29, 'intensity': 83.18},
    {'seq_len': 131, 'gflops': 11420.49, 'intensity': 161.84},
    {'seq_len': 259, 'gflops': 16641.87, 'intensity': 316.83},
    {'seq_len': 515, 'gflops': 21679.18, 'intensity': 617.97}
]

vit_data = [
    {'batch_size': 1, 'gflops': 2194.58, 'intensity': 20.25},
    {'batch_size': 8, 'gflops': 4151.03, 'intensity': 160.04},
    {'batch_size': 16, 'gflops': 4144.01, 'intensity': 315.71},
    {'batch_size': 32, 'gflops': 4118.97, 'intensity': 614.68},
    {'batch_size': 64, 'gflops': 3957.65, 'intensity': 1167.44}
]

resnet_data = [
    {'batch_size': 1, 'gflops': 467.48, 'intensity': 38.90},
    {'batch_size': 8, 'gflops': 4552.08, 'intensity': 298.85},
    {'batch_size': 16, 'gflops': 3598.57, 'intensity': 571.81},
    {'batch_size': 32, 'gflops': 3032.73, 'intensity': 1052.43},
    {'batch_size': 64, 'gflops': 2674.34, 'intensity': 1815.35}
]

plt.figure(figsize=(14, 10))

x_range = np.logspace(-1, 4, 1000)  

memory_bound = [x * memory_bw_gb for x in x_range]

compute_bound_fp32 = [peak_fp32_gflops for _ in x_range]
compute_bound_fp16 = [peak_fp16_gflops for _ in x_range]

plt.loglog(x_range, memory_bound, 'k-', label='Memory Bandwidth Limit', linewidth=2)
plt.loglog(x_range, compute_bound_fp32, 'r--', label='FP32 Compute Limit', linewidth=2)
plt.loglog(x_range, compute_bound_fp16, 'g--', label='FP16 Compute Limit', linewidth=2)

# OPT - Blue
for d in opt_data:
    plt.loglog(d['intensity'], d['gflops'], 'bo', markersize=10,
               label='OPT' if d == opt_data[0] else "")
    plt.annotate(f"Seq={d['seq_len']}",
                 (d['intensity'], d['gflops']),
                 xytext=(5, 5), textcoords='offset points', color='blue')

# ViT - Green
for d in vit_data:
    plt.loglog(d['intensity'], d['gflops'], 'go', markersize=10,
               label='ViT' if d == vit_data[0] else "")
    plt.annotate(f"BS={d['batch_size']}",
                 (d['intensity'], d['gflops']),
                 xytext=(5, 5), textcoords='offset points', color='green')

# ResNet50 - Red
for d in resnet_data:
    plt.loglog(d['intensity'], d['gflops'], 'ro', markersize=10,
               label='ResNet50' if d == resnet_data[0] else "")
    plt.annotate(f"BS={d['batch_size']}",
                 (d['intensity'], d['gflops']),
                 xytext=(5, 5), textcoords='offset points', color='red')

plt.xlabel('FLOPS/Byte', fontsize=12)
plt.ylabel('GFLOPS/s', fontsize=12)
plt.title('Roofline OPT, ViT, and ResNet50 on NVIDIA L4', fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(loc='lower right')

plt.annotate(f"FP32 Peak: {peak_fp32_performance} TFLOPS",
             (100, peak_fp32_gflops),
             xytext=(10, 10), textcoords='offset points')
plt.annotate(f"FP16 Peak: {peak_fp16_performance} TFLOPS",
             (100, peak_fp16_gflops),
             xytext=(10, 10), textcoords='offset points')
plt.annotate(f"Memory BW: {memory_bandwidth} TB/s",
             (1, memory_bw_gb),
             xytext=(10, 10), textcoords='offset points')

plt.tight_layout()
plt.savefig('multi_model_roofline.png', dpi=300)
plt.show()
