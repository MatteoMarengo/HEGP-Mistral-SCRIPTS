# app.py

import os
import chainlit as cl
from ctransformers import AutoModelForCausalLM,AutoConfig

model_name= "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
# path to the model file (local)
model_path = r"HEGP-Mistral-MODELS/GGUF-Mistral7B/mistral-7b-instruct-v0.1.Q5_K_M.gguf"

# Runs when the chat starts
@cl.on_chat_start
def main():
    # Create the llm
    config = AutoConfig.from_pretrained(model_name)
    # config.max_new_tokens = 2048
    config.config.context_length = 4096
    llm = AutoModelForCausalLM.from_pretrained(model_name,
                                               model_file=model_path,
                                               model_type="mistral",
                                               temperature=0.7,
                                               gpu_layers=0,
                                               stream=True,
                                               threads=int(os.cpu_count() ),
                                               max_new_tokens=200,
                                               config=config)

    # Store the llm in the user session
    cl.user_session.set("llm", llm)

# Runs when a message is sent
@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain from the user session
    llm = cl.user_session.get("llm")

    msg = cl.Message(
        content="",
    )

    prompt = f"[INST]{message.content}[/INST]"
    for text in llm(prompt=prompt):
        await msg.stream_token(text)

    await msg.send()