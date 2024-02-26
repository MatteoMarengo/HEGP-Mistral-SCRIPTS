import re

def extract_after_number(text):
    """
    Extract the answer from the prompt.
    Just the answer and not the "A1. " part.
    In the final version, just access the answers patient directly
    """
    match = re.search(r"A\d+\.\s*", text)
    if match:
        # Extract the position where the matched pattern ends
        end_pos = match.end()
        # Return the substring after this position
        return text[end_pos:]
    else:
        # If the pattern is not found, handle accordingly
        return "Pattern not found"

def define_keywords():
    """
    Define the keywords for each question based on the one provided in the questionnaire.
    """

    keywords = ["", "fatigue", "fatigue", "flatulence", "diarrhea", "abdominal abdomen pain", "abdominal abdomen pain", "abdominal abdomen pain",
    "urination pain", "urge urges urinate", "urge urges urinate", "frequent frequency urination", "urination urine frequency", "urine color coloration", "leakage", "leakage", "skin burns", ""]  
    return keywords

def define_scores_dict():
    """
    Just a simple scoring of the answers based on the gravity
    """
    scores_dict = {"yes":1, "no":0, 
                "not at all": 0, "a little bit":0.25, "somewhat":0.5, "quite a bit":0.75, "very much":1, 
                "never":0,"rarely":0.25, "occasionally":0.5, "frequently":0.75, "almost constantly":1, 
                "none":0, "mild":0.25, "moderate":0.5, "severe":0.75, "very severe":1,
                "not applicable": 0}
    return scores_dict

def answers_to_score(answers, scores_dict):
    """
    This method takes the answers and output a list of scores based on the scores dictionary.
    """
    adjusted_scores = {k.lower(): v for k, v in scores_dict.items()}
    scored_answers = []
    for answer in answers:
        extracted_answer = extract_after_number(answer).lower().strip()  # Ensure matching format
        print(extracted_answer)
        # extracted_answer = answer.lower().strip()  # Ensure matching format
        score = adjusted_scores.get(extracted_answer, 0)  # Default to 0 if not found
        print(score)
        scored_answers.append(score)
    scores = {i+1: value for i, value in enumerate(scored_answers)}
    print(scores)
    return scores

def check_presence(output, keyword):
  """
  This method checks the presence of the keyword in the summary.
  If there are several keywords, it checks that the majority (>= 0.5) is included in the summary
  """

  if " " in keyword:
    words = keyword.split()
    present_count = sum(word in output for word in words)
    return present_count >= len(words) / 2
  else:
    return keyword in output

def check_daily_activities_question(output, keyword):
  """
  This method checks the presence of the symptoms related to interfence in the daily activities.
  We specifically ask in the prompt a sentence beginning by "In his daily activities" followed by the most important symptoms
  """

  sentences = output.split(".")
  starting_phrase = "in his daily activities"

  # Check each sentence for the conditions
  for sentence in sentences:
    # Check if the sentence starts with the specific phrase and contains the word
    if sentence.lower().strip().startswith(starting_phrase):
      return check_presence(output, keyword)
  return False

def check_symptoms_question(output, keyword):
  """
  Same methods but for just normal syndroms.
  """
  output = output.lower()
  return check_presence(output, keyword)

def scores_properties():
    """
    Get the index of the symptoms and daily activities questions and transform keywords in a dictionary
    """
    
    keywords = define_keywords()
    keywords_dict = {i+1: value for i, value in enumerate(keywords)}
    symptoms = [2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17]
    daily_activities = [3, 8, 11, 13, 16]
    return keywords_dict, symptoms, daily_activities

def compute_metric(scores, keywords_dict, output, daily_activities, symptoms):
  """
  Compute the metric. When the keyword is present it increments by 1.
  The metric (for the moment) is just the ratio of all the important (score > 0.5) keywords in the summary / all the important keyword in the patient answers
  """
  total_sum, in_summary = 0, 0
  for key, value in scores.items():
    if value > 0.5:
      total_sum += 1
      if key in daily_activities:
        if check_daily_activities_question(output, keywords_dict[key]):
          in_summary += 1
      if key in symptoms:
        if check_symptoms_question(output, keywords_dict[key]):
          in_summary += 1

  if total_sum == 0: # No symptom graded > 0.5, just return 1
    return 1

  final_grade = in_summary / total_sum
  return final_grade
