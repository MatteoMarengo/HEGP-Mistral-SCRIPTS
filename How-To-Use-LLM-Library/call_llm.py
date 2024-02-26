#####
# Authors: LEVY Jarod & MARENGO Matteo
# Date: January 2024
# MVA ENS Paris-Saclay - 2023/2024
#####

#############################################################################################
# Import the subprocess module to run the command
import subprocess
import time
import sys
import csv
import pandas as pd
from process_QA import prompt_creation, random_prompt_creation, random_prompt_creation_with_csv

#############################################################################################
# Define the lists to store the output and elapsed time
output_list = []
elapsed_time_list = []

#############################################################################################
def main(prompt):
    # Define the command to be executed
    command = [
        "llm", "-m", "mistral-7b-instruct-v0", 
        prompt
    ]

    start_time = time.time()

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout, stderr = process.communicate()

    elapsed_time = time.time() - start_time

    # Save the output to a text file
    with open("output.txt", "w") as file:
        try:
            output = stdout.decode('utf-8')
            print("Output:\n", output)
            file.write(output)
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

    # Append output and elapsed time to the lists
    output_list.append(output)
    elapsed_time_list.append(elapsed_time)

       
#############################################################################################
# Run the main function
if __name__ == "__main__":
    num_generated_prompts = 1
    for i in range(num_generated_prompts):
        prompt,csv_file_path, excel_file_path = random_prompt_creation_with_csv(num_generated_prompts)
        main(prompt)

        with open(csv_file_path, "a", newline="") as file:
            writer = csv.writer(file)

            # Write the output list to the CSV file
            writer.writerow(['Generated Summary'] + output_list)

            # Write the elapsed time list to the CSV file
            writer.writerow(['Elapsed Time'] + elapsed_time_list)

        # Specify the path to your CSV file
        csv_file = 'HEGP-Mistral-SCRIPTS\generated_data\questions_and_answers.csv'

        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file_path)

        # Write the entire DataFrame to the Excel file
        df.to_excel(excel_file_path, index=False)

        


