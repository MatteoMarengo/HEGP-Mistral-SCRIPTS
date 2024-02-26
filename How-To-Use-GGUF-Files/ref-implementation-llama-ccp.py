from ctransformers import AutoModelForCausalLM, AutoConfig
import os
model_name= "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
model_path = r"HEGP-Mistral-MODELS/GGUF-Mistral7B/mistral-7b-instruct-v0.1.Q5_K_M.gguf"

# Create the llm
config = AutoConfig.from_pretrained(model_name)
# config.max_new_tokens = 2048
config.config.context_length = 4096
llm = AutoModelForCausalLM.from_pretrained(model_name
                                            model_file=model_path,
                                            model_type="mistral",
                                            temperature=0.7,
                                            gpu_layers=0,
                                            stream=True,
                                            threads=int(os.cpu_count() / 2),
                                            max_new_tokens=10000,
                                            config=config)
# text = "Q1. How many radiation treatments have you had? It’s okay if you don’t know. A1. 3 Q2. In the last 7 days, what was the SEVERITY of your FATIGUE, TIREDNESS, OR LACK OF ENERGY at its WORST? A2. Severe Q3. In the last 7 days, how much did FATIGUE, TIREDNESS, OR LACK OF ENERGY INTERFERE with your usual or daily activities? A3. Quite a bit Q4. In the last 7 days, did you have any INCREASED PASSING OF GAS (FLATULENCE)? A4. Yes Q5. In the last 7 days, how OFTEN did you have LOOSE OR WATERY STOOLS (DIARRHEA)? A5. Frequently Q6. In the last 7 days, how OFTEN did you have PAIN IN THE ABDOMEN (BELLY AREA)? A6. Frequently Q7. In the last 7 days, what was the SEVERITY of your PAIN IN THE ABDOMEN (BELLY AREA) at its WORST? A7. Severe Q8. In the last 7 days, how much did PAIN IN THE ABDOMEN (BELLY AREA) INTERFERE with your usual or daily activities? A8. Very much Q9. In the last 7 days, what was the SEVERITY of your PAIN OR BURNING WITH URINATION at its WORST? A9. Moderate Q10. In the last 7 days, how OFTEN did you feel an URGE TO URINATE ALL OF A SUDDEN? A10. Rarely Q11. In the last 7 days, how much did SUDDEN URGES TO URINATE INTERFERE with your usual or daily activities? A11. A little bit Q12. In the last 7 days, were there times when you had to URINATE FREQUENTLY? A12. Frequently Q13. In the last 7 days, how much did FREQUENT URINATION INTERFERE with your usual or daily activities? A13. Quite a bit Q14. In the last 7 days, did you have any URINE COLOR CHANGE? A14. No Q15. In the last 7 days, how OFTEN did you have LOSS OF CONTROL OF URINE (LEAKAGE)? A15. Occasionally Q16. In the last 7 days, how much did LOSS OF CONTROL OF URINE (LEAKAGE) INTERFERE with your usual or daily activities? A16. Somewhat Q17. In the last 7 days, what was the SEVERITY of your SKIN BURNS FROM RADIATION at their WORST? A17. Very Severe Q18. Finally, do you have any other symptoms that you wish to report? A18. Slight nausea and dizziness. You are an experienced radiation oncologist physician. You are provided this list of questions and answers about patient symptoms during their weekly follow-up visit during radiotherapy. Please summarize the following data into two sentences of natural language for your physician colleagues. Please put the most important symptoms first. Provide the summarization in the english language. Example: This patient with 7 radiation treatments is having severe abdominal pain, moderately affecting activities of daily living. Other symptoms include occasional diarrhea, mild rash. "
text = "Give me the recipe of a good french recipe"
prompt = f"[INST]{text}[/INST]"

# Generate the summary
print(llm(prompt=prompt))

from llama_cpp import Llama

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = Llama(
  model_path="./gemma-7b-GGUF.Q4_K_M.gguf",  # Download the model file first
  n_ctx=32768,  # The max sequence length to use - note that longer sequence lengths require much more resources
  n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
  n_gpu_layers=35         # The number of layers to offload to GPU, if you have GPU acceleration available
)

# Simple inference example
output = llm(
  "<|im_start|>system{system_message}<|im_end|><|im_start|>user{prompt}<|im_end|><|im_start|>assistant", # Prompt
  max_tokens=512,  # Generate up to 512 tokens
  stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
  echo=True        # Whether to echo the prompt
)

# Chat Completion API

llm = Llama(model_path="./gemma-7b-GGUF.Q4_K_M.gguf", chat_format="llama-2")  # Set chat_format according to the model you are using
llm.create_chat_completion(
    messages = [
        {"role": "system", "content": "You are a story writing assistant."},
        {
            "role": "user",
            "content": "Write a story about llamas."
        }
    ]
)


