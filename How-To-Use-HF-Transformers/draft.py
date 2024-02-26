from transformers import AutoModelForCausalLM, AutoTokenizer
# from process_QA import prompt_creation, random_prompt_creation
import torch  # Import torch
import time
import platform

if torch.cuda.is_available():
    print("CUDA is available. GPU can be used for computation.")
else:
    print("CUDA is not available. Using CPU for computation.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"  # Force to use CPU
print(f"Using device: {device}")

print("CPU Information:")
print(platform.processor())
import psutil

# Get the total physical memory (RAM)
total_memory = psutil.virtual_memory().total

# Convert bytes to GB for easier readability
total_memory_gb = total_memory / (1024**3)

print(f"Total RAM: {total_memory_gb:.2f} GB")

import GPUtil

gpus = GPUtil.getGPUs()
for gpu in gpus:
    print(f"GPU: {gpu.name}, Total VRAM: {gpu.memoryTotal}MB")




## Define the model and load it
model_name_original = "mistralai/Mistral-7B-v0.1"  # Replace with your model of choice
cache_dir = "D:/OneDrive/Documents/MVA-ENS-2023-2024/S1/HEGP/HEGP-Mistral-SCRIPTS/mistralai/Mistral-7B-v0.1"  # Replace with your desired path
# tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
# model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

token_name = "D:/OneDrive/Documents/MVA-ENS-2023-2024/S1/HEGP/HEGP-Mistral-SCRIPTS/models--mistralai--Mistral-7B-v0.1/snapshots/26bca36bde8333b5d7f72e9ed20ccda6a618af24"
# model_name = "D:/OneDrive/Documents/MVA-ENS-2023-2024/S1/HEGP/HEGP-Mistral-SCRIPTS/models--mistralai--Mistral-7B-v0.1"
print("loading the tokenizer ... ")
tokenizer = AutoTokenizer.from_pretrained(token_name)
print("tokenizer successfully loaded ! ")
print("loading the model ...")
model = AutoModelForCausalLM.from_pretrained(token_name)
print("model successfully loaded !")
# Move the model to CPU
model.to(device)

## Do a prompt test
prompt = "Hello my name is "
print(prompt)
model_inputs = tokenizer([prompt], return_tensors="pt")

# Move the model inputs to CPU
model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
output = tokenizer.batch_decode(generated_ids)[0]
print(output)
