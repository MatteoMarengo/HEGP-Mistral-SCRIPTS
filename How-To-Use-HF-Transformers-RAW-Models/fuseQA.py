from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-v0.1"  # Replace with your model of choice
cache_dir = "D:/OneDrive/Documents/MVA-ENS-2023-2024/S1/HEGP/Mistral7B-scripts"  # Replace with your desired path

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

prompt = "Hello my name is "
model_inputs = tokenizer([prompt], return_tensors="pt")
generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
tokenizer.batch_decode(generated_ids)[0]


