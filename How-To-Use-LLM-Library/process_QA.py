#####
# Authors: LEVY Jarod & MARENGO Matteo
# Date: January 2024
# MVA ENS Paris-Saclay - 2023/2024
#####

#############################################################################################
# Import libraries
import numpy as np
import random
import csv
import os

#############################################################################################
# Function to select and format answers
def format_answers_randomly(answers):
    formatted_answers = []
    for i, answer_options in enumerate(answers, 1):
        random_answer = random.choice(answer_options)
        formatted_answer = f"A{i+1}. {random_answer} "
        formatted_answers.append(formatted_answer)
    return formatted_answers

#############################################################################################
# Function to create a prompt with random answers
def random_prompt_creation():
        # List of lists containing possible answers for each question
    
    questions = [
        "Q1. How many radiation treatments have you had? It’s okay if you don’t know. ",
        "Q2. In the last 7 days, what was the SEVERITY of your FATIGUE, TIREDNESS, OR LACK OF ENERGY at its WORST? ",
        "Q3. In the last 7 days, how much did FATIGUE, TIREDNESS, OR LACK OF ENERGY INTERFERE with your usual or daily activities? ",
        "Q4. In the last 7 days, did you have any INCREASED PASSING OF GAS (FLATULENCE)? ",
        "Q5. In the last 7 days, how OFTEN did you have LOOSE OR WATERY STOOLS (DIARRHEA)? ",
        "Q6. In the last 7 days, how OFTEN did you have PAIN IN THE ABDOMEN (BELLY AREA)? ",
        "Q7. In the last 7 days, what was the SEVERITY of your PAIN IN THE ABDOMEN (BELLY AREA) at its WORST? ",
        "Q8. In the last 7 days, how much did PAIN IN THE ABDOMEN (BELLY AREA) INTERFERE with your usual or daily activities? ",
        "Q9. In the last 7 days, what was the SEVERITY of your PAIN OR BURNING WITH URINATION at its WORST? ",
        "Q10. In the last 7 days, how OFTEN did you feel an URGE TO URINATE ALL OF A SUDDEN? ",
        "Q11. In the last 7 days, how much did SUDDEN URGES TO URINATE INTERFERE with your usual or daily activities? ",
        "Q12. In the last 7 days, were there times when you had to URINATE FREQUENTLY? ",
        "Q13. In the last 7 days, how much did FREQUENT URINATION INTERFERE with your usual or daily activities? ",
        "Q14. In the last 7 days, did you have any URINE COLOR CHANGE? ",
        "Q15. In the last 7 days, how OFTEN did you have LOSS OF CONTROL OF URINE (LEAKAGE)? ",
        "Q16. In the last 7 days, how much did LOSS OF CONTROL OF URINE (LEAKAGE) INTERFERE with your usual or daily activities? ",
        "Q17. In the last 7 days, what was the SEVERITY of your SKIN BURNS FROM RADIATION at their WORST? ",
        "Q18. Finally, do you have any other symptoms that you wish to report? "
    ]

    possible_answers = [
        ["None", "Mild", "Moderate", "Severe", "Very Severe"],  # A2
        ["Not at all", "A little bit", "Somewhat", "Quite a bit"],  # A3
        ["Yes", "No"],  # A4
        ["Never", "Rarely", "Occasionally", "Frequently", "Almost constantly"],  # A5
        ["Never", "Rarely", "Occasionally", "Frequently", "Almost Constantly"],  # A6
        ["None", "Mild", "Moderate", "Severe", "Very Severe"],  # A7
        ["Not at all", "A little bit", "Somewhat", "Quite a bit", "Very much"],  # A8
        ["None", "Mild", "Moderate", "Severe", "Very Severe"],  # A9
        ["Never", "Rarely", "Occasionally", "Frequently", "Almost Constantly"],  # A10
        ["Not at all", "A little bit", "Somewhat", "Quite a bit", "Very much"],  # A11
        ["Never", "Rarely", "Occasionally", "Frequently", "Almost Constantly"],  # A12
        ["Not at all", "A little bit", "Somewhat", "Quite a bit", "Very much"],  # A13
        ["Yes", "No"],  # A14
        ["Never", "Rarely", "Occasionally", "Frequently", "Almost Constantly"],  # A15
        ["Not at all", "A little bit", "Somewhat", "Quite a bit", "Very much"],  # A16
        ["None", "Mild", "Moderate", "Severe", "Very Severe", "Not Applicable"],  # A17
        ["No additional symptoms", "Experiencing occasional headaches", "Slight nausea and dizziness"]  # A18
    ]
    answers = []
    answers.append("A1. " + str(random.randint(1, 15)))
    answers += format_answers_randomly(possible_answers)

    instructions = "You are an experienced radiation oncologist physician. You are provided this list of questions and answers about patient symptoms during their weekly follow-up visit during radiotherapy. Please summarize the following data into two sentences of natural language for your physician colleagues. Please put the most important symptoms first. Provide the summarization in the english language. "
    example = "Example: This patient with 7 radiation treatments is having severe abdominal pain, moderately affecting activities of daily living. Other symptoms include occasional diarrhea, mild rash. "
    prompt = ""
    
    language = "French"
    instruction_language = f"Please provide the answer in {language}"


    for i in range(len(questions)):
        prompt += questions[i] 
        prompt += answers[i]

    prompt = prompt + instructions + example + instruction_language

    return prompt

#############################################################################################
# prompt = "Q1. How many radiation treatments have you had? It’s okay if you don’t know. A1. 3 Q2. In the last 7 days, what was the SEVERITY of your FATIGUE, TIREDNESS, OR LACK OF ENERGY at its WORST? A2. Severe Q3. In the last 7 days, how much did FATIGUE, TIREDNESS, OR LACK OF ENERGY INTERFERE with your usual or daily activities? A3. Quite a bit Q4. In the last 7 days, did you have any INCREASED PASSING OF GAS (FLATULENCE)? A4. Yes Q5. In the last 7 days, how OFTEN did you have LOOSE OR WATERY STOOLS (DIARRHEA)? A5. Frequently Q6. In the last 7 days, how OFTEN did you have PAIN IN THE ABDOMEN (BELLY AREA)? A6. Frequently Q7. In the last 7 days, what was the SEVERITY of your PAIN IN THE ABDOMEN (BELLY AREA) at its WORST? A7. Severe Q8. In the last 7 days, how much did PAIN IN THE ABDOMEN (BELLY AREA) INTERFERE with your usual or daily activities? A8. Very much Q9. In the last 7 days, what was the SEVERITY of your PAIN OR BURNING WITH URINATION at its WORST? A9. Moderate Q10. In the last 7 days, how OFTEN did you feel an URGE TO URINATE ALL OF A SUDDEN? A10. Rarely Q11. In the last 7 days, how much did SUDDEN URGES TO URINATE INTERFERE with your usual or daily activities? A11. A little bit Q12. In the last 7 days, were there times when you had to URINATE FREQUENTLY? A12. Frequently Q13. In the last 7 days, how much did FREQUENT URINATION INTERFERE with your usual or daily activities? A13. Quite a bit Q14. In the last 7 days, did you have any URINE COLOR CHANGE? A14. No Q15. In the last 7 days, how OFTEN did you have LOSS OF CONTROL OF URINE (LEAKAGE)? A15. Occasionally Q16. In the last 7 days, how much did LOSS OF CONTROL OF URINE (LEAKAGE) INTERFERE with your usual or daily activities? A16. Somewhat Q17. In the last 7 days, what was the SEVERITY of your SKIN BURNS FROM RADIATION at their WORST? A17. Very Severe Q18. Finally, do you have any other symptoms that you wish to report? A18. Slight nausea and dizziness.You are an experienced radiation oncologist physician. You are provided this list of questions and answers about patient symptoms during their weekly follow-up visit during radiotherapy. Please summarize the following data into two sentences of natural language for your physician colleagues. Please put the most important symptoms first. Provide the summarization in the english language. Example: This patient with 7 radiation treatments is having severe abdominal pain, moderately affecting activities of daily living. Other symptoms include occasional diarrhea, mild rash."
def prompt_creation():

    questions = [
        "Q1. How many radiation treatments have you had? It’s okay if you don’t know. ",
        "Q2. In the last 7 days, what was the SEVERITY of your FATIGUE, TIREDNESS, OR LACK OF ENERGY at its WORST? ",
        "Q3. In the last 7 days, how much did FATIGUE, TIREDNESS, OR LACK OF ENERGY INTERFERE with your usual or daily activities? ",
        "Q4. In the last 7 days, did you have any INCREASED PASSING OF GAS (FLATULENCE)? ",
        "Q5. In the last 7 days, how OFTEN did you have LOOSE OR WATERY STOOLS (DIARRHEA)? ",
        "Q6. In the last 7 days, how OFTEN did you have PAIN IN THE ABDOMEN (BELLY AREA)? ",
        "Q7. In the last 7 days, what was the SEVERITY of your PAIN IN THE ABDOMEN (BELLY AREA) at its WORST? ",
        "Q8. In the last 7 days, how much did PAIN IN THE ABDOMEN (BELLY AREA) INTERFERE with your usual or daily activities? ",
        "Q9. In the last 7 days, what was the SEVERITY of your PAIN OR BURNING WITH URINATION at its WORST? ",
        "Q10. In the last 7 days, how OFTEN did you feel an URGE TO URINATE ALL OF A SUDDEN? ",
        "Q11. In the last 7 days, how much did SUDDEN URGES TO URINATE INTERFERE with your usual or daily activities? ",
        "Q12. In the last 7 days, were there times when you had to URINATE FREQUENTLY? ",
        "Q13. In the last 7 days, how much did FREQUENT URINATION INTERFERE with your usual or daily activities? ",
        "Q14. In the last 7 days, did you have any URINE COLOR CHANGE? ",
        "Q15. In the last 7 days, how OFTEN did you have LOSS OF CONTROL OF URINE (LEAKAGE)? ",
        "Q16. In the last 7 days, how much did LOSS OF CONTROL OF URINE (LEAKAGE) INTERFERE with your usual or daily activities? ",
        "Q17. In the last 7 days, what was the SEVERITY of your SKIN BURNS FROM RADIATION at their WORST? ",
        "Q18. Finally, do you have any other symptoms that you wish to report? "
    ]

    answers = [
        "A1. 7 ",
        "A2. Mild ",
        "A3. Not at all ",
        "A4. Yes ",
        "A5. Rarely ",
        "A6. Occasionally ",
        "A7. Mild ",
        "A8. A little bit ",
        "A9. None ",
        "A10. Occasionally ",
        "A11. Not at all ",
        "A12. Rarely ",
        "A13. Not at all ",
        "A14. No ",
        "A15. Never ",
        "A16. Not at all ",
        "A17. Mild ",
        "A18. No additional symptoms. "
    ]

    instructions = "You are an experienced radiation oncologist physician. You are provided this list of questions and answers about patient symptoms during their weekly follow-up visit during radiotherapy. Please summarize the following data into two sentences of natural language for your physician colleagues. Please put the most important symptoms first. Provide the summarization in the english language. "
    example = "Example: This patient with 7 radiation treatments is having severe abdominal pain, moderately affecting activities of daily living. Other symptoms include occasional diarrhea, mild rash. "
    prompt = ""

    for i in range(len(questions)):
        prompt += questions[i] 
        prompt += answers[i]

    prompt = prompt + instructions + example

    return prompt

#############################################################################################
# Function to create a prompt with random answers and save it in a csv file
def random_prompt_creation_with_csv(num_generated_prompts):
  
    questions = [
        "Q1. How many radiation treatments have you had? It’s okay if you don’t know. ",
        "Q2. In the last 7 days, what was the SEVERITY of your FATIGUE, TIREDNESS, OR LACK OF ENERGY at its WORST? ",
        "Q3. In the last 7 days, how much did FATIGUE, TIREDNESS, OR LACK OF ENERGY INTERFERE with your usual or daily activities? ",
        "Q4. In the last 7 days, did you have any INCREASED PASSING OF GAS (FLATULENCE)? ",
        "Q5. In the last 7 days, how OFTEN did you have LOOSE OR WATERY STOOLS (DIARRHEA)? ",
        "Q6. In the last 7 days, how OFTEN did you have PAIN IN THE ABDOMEN (BELLY AREA)? ",
        "Q7. In the last 7 days, what was the SEVERITY of your PAIN IN THE ABDOMEN (BELLY AREA) at its WORST? ",
        "Q8. In the last 7 days, how much did PAIN IN THE ABDOMEN (BELLY AREA) INTERFERE with your usual or daily activities? ",
        "Q9. In the last 7 days, what was the SEVERITY of your PAIN OR BURNING WITH URINATION at its WORST? ",
        "Q10. In the last 7 days, how OFTEN did you feel an URGE TO URINATE ALL OF A SUDDEN? ",
        "Q11. In the last 7 days, how much did SUDDEN URGES TO URINATE INTERFERE with your usual or daily activities? ",
        "Q12. In the last 7 days, were there times when you had to URINATE FREQUENTLY? ",
        "Q13. In the last 7 days, how much did FREQUENT URINATION INTERFERE with your usual or daily activities? ",
        "Q14. In the last 7 days, did you have any URINE COLOR CHANGE? ",
        "Q15. In the last 7 days, how OFTEN did you have LOSS OF CONTROL OF URINE (LEAKAGE)? ",
        "Q16. In the last 7 days, how much did LOSS OF CONTROL OF URINE (LEAKAGE) INTERFERE with your usual or daily activities? ",
        "Q17. In the last 7 days, what was the SEVERITY of your SKIN BURNS FROM RADIATION at their WORST? ",
        "Q18. Finally, do you have any other symptoms that you wish to report? "
    ]

    possible_answers = [
        ["None", "Mild", "Moderate", "Severe", "Very Severe"],  # A2
        ["Not at all", "A little bit", "Somewhat", "Quite a bit"],  # A3
        ["Yes", "No"],  # A4
        ["Never", "Rarely", "Occasionally", "Frequently", "Almost constantly"],  # A5
        ["Never", "Rarely", "Occasionally", "Frequently", "Almost Constantly"],  # A6
        ["None", "Mild", "Moderate", "Severe", "Very Severe"],  # A7
        ["Not at all", "A little bit", "Somewhat", "Quite a bit", "Very much"],  # A8
        ["None", "Mild", "Moderate", "Severe", "Very Severe"],  # A9
        ["Never", "Rarely", "Occasionally", "Frequently", "Almost Constantly"],  # A10
        ["Not at all", "A little bit", "Somewhat", "Quite a bit", "Very much"],  # A11
        ["Never", "Rarely", "Occasionally", "Frequently", "Almost Constantly"],  # A12
        ["Not at all", "A little bit", "Somewhat", "Quite a bit", "Very much"],  # A13
        ["Yes", "No"],  # A14
        ["Never", "Rarely", "Occasionally", "Frequently", "Almost Constantly"],  # A15
        ["Not at all", "A little bit", "Somewhat", "Quite a bit", "Very much"],  # A16
        ["None", "Mild", "Moderate", "Severe", "Very Severe", "Not Applicable"],  # A17
        ["No additional symptoms", "Experiencing occasional headaches", "Slight nausea and dizziness"]  # A18
    ]
    answers = []
    answers.append("A1. " + str(random.randint(1, 15)))
    answers += format_answers_randomly(possible_answers)

    instructions = "You are an experienced radiation oncologist physician. You are provided this list of questions and answers about patient symptoms during their weekly follow-up visit during radiotherapy. Please summarize the following data into two sentences of natural language for your physician colleagues. Please put the most important symptoms first. Provide the summarization in the english language. "
    example = "Example: This patient with 7 radiation treatments is having severe abdominal pain, moderately affecting activities of daily living. Other symptoms include occasional diarrhea, mild rash. "
    prompt = ""
    
    language = "French"
    instruction_language = f"Please provide the answer in {language}"

    for i in range(len(questions)):
        prompt += questions[i] 
        prompt += answers[i]

    prompt = prompt + instructions + example + instruction_language
    csv_file_path = 'generated_data/questions_and_answers_'+str(num_generated_prompts)+'.csv'
    excel_file_path = 'generated_data/questions_and_answers_'+str(num_generated_prompts)+'.xlsx'

    # Check if CSV exists and read it
    if os.path.exists(csv_file_path):
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            existing_rows = list(reader)
            header = existing_rows[0]
            # Find the next available column
            next_available_col = len(header)
            # Update the header if a new column is needed
            if any(row[next_available_col-1] for row in existing_rows[1:]):
                header.append(f'Answer {next_available_col//2}')
    else:
        # If the file doesn't exist, create a new one with a header
        existing_rows = []
        header = ['Question', 'Answer 1']

    # Create or update the CSV file and write the data
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write the updated header
        writer.writerow(header)

        # Write the existing rows, and add new answers if the file already existed
        for i, row in enumerate(existing_rows[1:]):
            while len(row) < next_available_col:  # Pad rows if they are too short
                row.append('')
            row.append(answers[i].strip())  # Add the new answer
            writer.writerow(row)

        # If it's a new file, just write the questions and answers
        if not existing_rows:
            for question, answer in zip(questions, answers):
                writer.writerow([question.strip(), answer.strip()])

    return prompt,csv_file_path,excel_file_path
