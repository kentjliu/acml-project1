import os
os.environ["HF_TOKEN"] = "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"

import torch
import time
from transformers import LlamaForCausalLM, AutoTokenizer, LlamaConfig, AutoModelForCausalLM
from torch.profiler import profile, record_function, ProfilerActivity

model_name = "facebook/opt-6.7b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 

model = None

model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            # config=config,
            cache_dir=None,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )



model.eval()

prompt = "Columbia University is the best "
input_tokens = tokenizer(prompt, return_tensors="pt").to('cuda')

for _ in range(5):
    _ = model.generate(**input_tokens, max_new_tokens=1)

num_runs = 100
times = []
with torch.no_grad():
    for _ in range(num_runs):
        start_time = time.time()
        _ = model.generate(**input_tokens, max_new_tokens=1)
        torch.cuda.synchronize()
        times.append(time.time() - start_time)

avg_latency = sum(times) / num_runs 
throughput = 1 / avg_latency 

print(f"Avg Latency per Token: {avg_latency * 1000:.2f} ms")
print(f"Throughput: {throughput:.2f} tokens/sec")

gpu_memory = torch.cuda.memory_allocated('cuda') / (1024 ** 2)
print(f"GPU Memory Used: {gpu_memory:.2f} MB")

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
    with record_function("model_inference"):
        _ = model.generate(**input_tokens, max_new_tokens=1)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
