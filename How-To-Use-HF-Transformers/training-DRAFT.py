from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import MistralForCausalLM, LlamaTokenizer
import torch
import time
import platform
import transformers

transformers.logging.set_verbosity_info()

import torch
import psutil

# Check for available GPU(s) and their memory usage
if torch.cuda.is_available():
    print("CUDA is available.")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(i)/1024**3:.2f} GB")
        print(f"Cached:    {torch.cuda.memory_cached(i)/1024**3:.2f} GB\n")
else:
    print("CUDA is not available.")


# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# GPU information (optional)
if device == "cuda":
    print("GPU Information:")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU Compute Capability: {gpu.major}.{gpu.minor}")
    print(f"GPU Memory: {gpu.total_memory / 1024**3:.2f} GB")
# device = "cpu"

# CPU information (optional)
print("CPU Information:")
print(platform.processor())

ram = psutil.virtual_memory()
print(f"Available RAM on CPU: {ram.available / 1024**3:.2f} GB")

# GPU information (optional)
if device == "cuda":
    print("GPU Information:")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU Compute Capability: {gpu.major}.{gpu.minor}")
    print(f"GPU Memory: {gpu.total_memory / 1024**3:.2f} GB")

device = "cpu"
# Define the local directory where the model and tokenizer are stored
# local_model_directory = "D:/OneDrive/Documents/MVA-ENS-2023-2024/S1/HEGP/HEGP-Mistral-SCRIPTS/models--mistralai--Mistral-7B-v0.1/snapshots/26bca36bde8333b5d7f72e9ed20ccda6a618af24"
# local_model_directory = "D:/OneDrive/Documents/MVA-ENS-2023-2024/S1/HEGP/HEGP-Mistral-SCRIPTS/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/9ab9e76e2b09f9f29ea2d56aa5bd139e4445c59e"
#local_model_directory = "D:/OneDrive/Documents/MVA-ENS-2023-2024/S1/HEGP/HEGP-Mistral-SCRIPTS/models--mistralai--Mistral-7B-Instruct-v0.1"
# local_model_directory = "D:/OneDrive/Documents/MVA-ENS-2023-2024/S1/HEGP/HEGP-Mistral-SCRIPTS/Mistral-RAW"
local_model_directory = "D:/OneDrive/Documents/MVA-ENS-2023-2024/S1/HEGP/HEGP-Mistral-SCRIPTS/mistral-7b-instruct-v0.1.Q2_K.gguf"

# Load the tokenizer and model from the local directory
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(local_model_directory)
# tokenizer = LlamaTokenizer.from_pretrained(local_model_directory)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(local_model_directory)
# model = MistralForCausalLM.from_pretrained(local_model_directory)

# Move the model to the specified device (GPU or CPU)
model.to(device)

print("Model and tokenizer are loaded.")

# Do a prompt test
prompt = "Hello my name is "
print(prompt)
model_inputs = tokenizer([prompt], return_tensors="pt").to(device)

# Generate text
generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
output = tokenizer.batch_decode(generated_ids)[0]
print(output)
