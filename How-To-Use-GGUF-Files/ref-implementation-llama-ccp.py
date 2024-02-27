from llama_cpp import Llama
import os
import time
import json
import copy

start_time = time.time()

model_path = "D:/OneDrive/Documents/MVA-ENS-2023-2024/S1/HEGP/HEGP-Mistral-MODELS/GGUF-Mistral7B/mistral-7b-instruct-v0.1.Q5_K_M.gguf"
# model_path = "D:/OneDrive/Documents/MVA-ENS-2023-2024/S1/HEGP/HEGP-Mistral-MODELS/GGUF-BioMistral7B/ggml-model-Q5_K_M.gguf"

print("Loading model...")
llm = Llama(
  model_path=model_path,  # Download the model file first
  n_ctx=1500,  # The max sequence length to use - note that longer sequence lengths require much more resources
  n_threads=int(os.cpu_count()),            # The number of CPU threads to use, tailor to your system and the resulting performance
  n_gpu_layers=0
)
print("Model loaded")

prompt = "[INST]Q1. How many radiation treatments have you had? It’s okay if you don’t know.  ()A1. 9 Q2. In the last 7 days, what was the SEVERITY of your FATIGUE, TIREDNESS, OR LACK OF ENERGY at its WORST?  (fatigue)A2. Severe Q3. In the last 7 days, how much did FATIGUE, TIREDNESS, OR LACK OF ENERGY INTERFERE with your usual or daily activities?  (fatigue)A3. Not at all Q4. In the last 7 days, did you have any INCREASED PASSING OF GAS (FLATULENCE)?  (flatulence)A4. No Q5. In the last 7 days, how OFTEN did you have LOOSE OR WATERY STOOLS (DIARRHEA)?  (diarrhea)A5. Rarely Q6. In the last 7 days, how OFTEN did you have PAIN IN THE ABDOMEN (BELLY AREA)?  (abdominal abdomen pain)A6. Frequently Q7. In the last 7 days, what was the SEVERITY of your PAIN IN THE ABDOMEN (BELLY AREA) at its WORST?  (abdominal abdomen pain)A7. None Q8. In the last 7 days, how much did PAIN IN THE ABDOMEN (BELLY AREA) INTERFERE with your usual or daily activities?  (abdominal abdomen pain)A8. Quite a bit Q9. In the last 7 days, what was the SEVERITY of your PAIN OR BURNING WITH URINATION at its WORST?  (urination pain)A9. Very severe Q10. In the last 7 days, how OFTEN did you feel an URGE TO URINATE ALL OF A SUDDEN?  (urge urges urinate)A10. Occasionally Q11. In the last 7 days, how much did SUDDEN URGES TO URINATE INTERFERE with your usual or daily activities?  (urge urges urinate)A11. Somewhat Q12. In the last 7 days, were there times when you had to URINATE FREQUENTLY?  (frequent frequency urination)A12. Almost constantly Q13. In the last 7 days, how much did FREQUENT URINATION INTERFERE with your usual or daily activities?  (urination urine frequency)A13. A little bit Q14. In the last 7 days, did you have any URINE COLOR CHANGE?  (urine color coloration)A14. No Q15. In the last 7 days, how OFTEN did you have LOSS OF CONTROL OF URINE (LEAKAGE)?  (leakage)A15. Occasionally Q16. In the last 7 days, how much did LOSS OF CONTROL OF URINE (LEAKAGE) INTERFERE with your usual or daily activities?  (leakage)A16. Very much Q17. In the last 7 days, what was the SEVERITY of your SKIN BURNS FROM RADIATION at their WORST?  (skin burns)A17. Moderate Q18. Finally, do you have any other symptoms that you wish to report?  ()A18. No You are an experienced radiation oncologist physician. You are provided this list of questions and answers about patient symptoms during their weekly follow-up visit during radiotherapy. Please summarize the following data into two sentences of natural language for your physician colleagues. Indicate in your summary the most important symptoms using exactly the group of words in parenthesis at the end of the question. A “yes” is important. Each time, include the number of radiation treatments and the answer of the last question if answered. Treat the question including “daily activities” separately with a sentence beginning by - In his daily activities - and take only the important symptoms.Example: This patient, after 30 radiation treatments, reports very severe symptoms including fatigue, flatulence and diarrhea with occasional fever as an additional symptom. In his daily activities, very severe interference was noted due to fatigue and abdominal pain.Please provide the answer in English[/INST]"

print("Running model...")
output = llm(
    prompt = prompt,
    max_tokens=400,
    temperature=0.7,
    echo=True
)

# prompt = "How are you ?"
# print("Running model...")
# output = llm(
#     prompt = prompt,
#     max_tokens=200,
#     temperature=0.2,
#     echo=True
# )

print("--- %s seconds ---" % (time.time() - start_time))
print(output)
print(json.dumps(output, indent=2))
outputj = copy.deepcopy(output)
txt = outputj["choices"][0]["text"]


# Define the question text you want to remove, including any trailing new lines or spaces

# Remove the question from the response
answer_only = txt.replace(prompt, "", 1) # The '1' ensures only the first occurrence is replaced
# Remove any leading or trailing new lines or spaces
answer_only = answer_only.strip()
print(answer_only)

