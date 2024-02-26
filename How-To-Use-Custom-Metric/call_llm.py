import subprocess
import time
import re

from process_QA import prompt_creation, random_prompt_creation
from evaluation import compute_metric, scores_properties


"""
    prompt, scores = random_prompt_creation()
    with open(f"prompt.txt", "w") as file:
        file.write(prompt)
        file.write(str(scores))
    filename = f"output.txt"
    main(prompt, scores, filename)

"""

def main(prompt, scores, filename):
    # Define the command to be executed    
    command = [
        "llm", "-m", "nous-hermes-llama2-13b", 
        prompt
    ]

    start_time = time.time()

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout, stderr = process.communicate()

    elapsed_time = time.time() - start_time

    # Save the output to a text file
    with open(filename, "w") as file:
        try:
            output = stdout.decode('utf-8')
            keywords_dict, symptoms, daily_activities = scores_properties()
            grade = compute_metric(scores, keywords_dict, output, daily_activities, symptoms)
            print("Output:\n", output)
            file.write(output)
            file.write(f"Metric= {grade:.2f}\n")
        except UnicodeDecodeError:
            output = stdout.decode('utf-8', 'backslashreplace')
            print("Output (with undecodable bytes):\n", output)
            file.write(output)

        if stderr:
            try:
                error = stderr.decode('utf-8')
                print("Error:\n", error)
                file.write("\nError:\n" + error)
            except UnicodeDecodeError:
                error = stderr.decode('utf-8', 'backslashreplace')
                print("Error (with undecodable bytes):\n", error)
                file.write("\nError (with undecodable bytes):\n" + error)
        else:
            print("No error detected")
            file.write("No error detected")

    print(f"Time taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    # Outputting 10 different prompts 
    for i in range(1):
        prompt, scores = random_prompt_creation()
        with open(f"prompt_{i}.txt", "w") as file:
            file.write(prompt)
            file.write(str(scores))
        filename = f"output_{i}.txt"
        main(prompt, scores, filename)


