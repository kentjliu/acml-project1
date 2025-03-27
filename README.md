# Background
With the recent advancement of technology in AI, we have access to more choices than ever when it comes to both hardware and models.  With this abundance, we are privileged to consider trade-offs based on performance, workload, and resource availability. In this report, I analyze the performance of select AI models across three different computing environments, providing insights into how hardware choices impact inference speed, computational efficiency, and overall feasibility. By analyzing the performance of select models across three different environments, we are able to gain some valuable insights about the choice of environment and models and potential tradeoffs for different workloads.

# Experiment Design
With appropriate permissions and credits, Google Colab allows access to a variety of GPU runtimes. With this, we can compare different NN models from different domains and their performance in each environment. I will profile throughput and memory and use the roofline model to compare performance. In addition, I conduct some preliminary runs to gather data on compute utilization, memory bandwidth, and latency. By analyzing these factors, we can understand how different models behave under various hardware constraints and identify potential bottlenecks in system performance.

## NVIDIA GPU Specs
| GPU   | Peak FP32 Performance (TFLOPS) | Peak FP16 Performance (TFLOPS) | Memory Bandwidth (TB/s) |
|-------|-------------------------------|-------------------------------|-------------------------|
| **A100** | 19.5                          | 312                           | 1.6                     |
| **T4**   | 8.1                           | 65                            | 0.3                     |
| **L4**   | 30.3                          | 242                           | 0.3                     |

Sources:
* A100: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf
* L4: https://resources.nvidia.com/en-us-data-center-overview/l4-gpu-datasheet 
* T4: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf

# PyTorch Profiler Data
### ResNet50
| Metric                 | T4    | L4    | A100  |
|------------------------|-------|-------|-------|
| **Avg Latency (ms)**   | 82.42 | 41.92 | 11.87 |
| **Throughput (img/s)** | 388.24 | 763.35 | 2694.75 |
| **GPU Memory (MB)**    | 124.47 | 124.47 | 124.35 |
| **Self CPU time (ms)** | 159.333 | 53.517 | 34.266 |
| **Self GPU time (ms)** | 74.446 | 40.094 | 11.460 |

**Description:** ResNet50 performance with batch size 32. Latency decreases and throughput increases with increased GPU power. Memory usage remains similar due to the fixed batch size.

---

### ViTb16
| Metric                 | T4    | L4    | A100  |
|------------------------|-------|-------|-------|
| **Avg Latency (ms)**   | 310.34 | 166.64 | 71.02 |
| **Throughput (img/s)** | 103.11 | 192.03 | 450.59 |
| **GPU Memory (MB)**    | 356.85 | 356.85 | 356.85 |
| **Self CPU time (ms)** | 329.951 | 166.828 | 88.754 |
| **Self GPU time (ms)** | 319.239 | 154.940 | 70.991 |

**Description:** ViTb16 performance with batch size 32. Latency decreases and throughput increases with increased GPU power. Memory usage remains similar due to the fixed batch size.

---

### OPT
| Metric                 | T4    | L4    | A100  |
|------------------------|-------|-------|-------|
| **Avg Latency (ms)**   | 63.11 | 57.36 | 23.12 |
| **Throughput (tok/s)** | 15.84 | 17.43 | 43.25 |
| **GPU Memory (MB)**    | 12708.16 | 12708.16 | 12708.16 |
| **Self CPU time (ms)** | 86.820 | 79.407 | 120.470 |
| **Self GPU time (ms)** | 59.276 | 54.937 | 13.388 |

**Description:** OPT model performance with the prompt _"Columbia University is the best"_. Latency decreases and throughput increases with increased GPU power. Memory usage remains similar due to the fixed batch size.

# Additional Results
Can be found in the folders under the `doc` directory.

