from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)

print(completion.choices[0].message)

# #%env OPENAI_API_KEY = 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

# # from openai import OpenAI
# import openai
# openai.api_key = "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"

# completion = openai.Completion()

# prompt = "Hello, how are you?"
# model = "text-davinci-003"
# temperature = 0.5 # Controls randomness: 0.0 means the model will always choose the highest probability output, 1.0 means the model will choose completely randomly between all possibilities
# max_tokens = 100 # Maximum number of tokens to generate
# stop="\n"

# response = completion.generate(
#     prompt = prompt,
#     model = model,
#     temperature = temperature,
#     max_tokens = max_tokens,
#     stop = stop
# )

# print(response.choices[0].text.strip()) # "Hello, I'm good thank you. How are you?"



# response = openai.ChatCompletion.create( model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Generate a 3 sentence story about friendship"}] ) 
# print(response)