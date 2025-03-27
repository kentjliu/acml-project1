import torch
import torchvision.models as models
import time
from torch.profiler import profile, record_function, ProfilerActivity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=False).to(device)
model.eval()

batch_size = 32
input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)

for _ in range(10):
    _ = model(input_tensor)

num_runs = 100
times = []
with torch.no_grad():
    for _ in range(num_runs):
        start_time = time.time()
        _ = model(input_tensor)
        torch.cuda.synchronize()
        times.append(time.time() - start_time)

avg_latency = sum(times) / num_runs
throughput = batch_size / avg_latency

print(f"Avg Latency: {avg_latency * 1000:.2f} ms")
print(f"Throughput: {throughput:.2f} images/sec")

gpu_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
print(f"GPU Memory Used: {gpu_memory:.2f} MB")

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        _ = model(input_tensor)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
