from transformers import AutoModelForCausalLM, AutoTokenizer
from process_QA import prompt_creation, random_prompt_creation
import time

## Define the model and load it
model_name = "mistralai/Mistral-7B-v0.1"  # Replace with your model of choice
cache_dir = "D:/OneDrive/Documents/MVA-ENS-2023-2024/S1/HEGP/Mistral7B-scripts"  # Replace with your desired path

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

## Do a prompt test
prompt = "Hello my name is "
model_inputs = tokenizer([prompt], return_tensors="pt")
generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
output = tokenizer.batch_decode(generated_ids)[0]
print(output)

# ## Do a prompt test with a random prompt and save the output
# prompt = random_prompt_creation()
# model_inputs = tokenizer([prompt], return_tensors="pt")
# generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
# output = tokenizer.batch_decode(generated_ids)[0]

# ## Save the output to a text file
# # First column is the questions, second column is the answers
# # On the last line of the file, we save the generated summary
# with open("output.txt", "w") as file:
#     file.write(prompt + "\n")
#     file.write(output + "\n")
#     file.write("Time taken: " + str(elapsed_time) + " seconds\n")
    




