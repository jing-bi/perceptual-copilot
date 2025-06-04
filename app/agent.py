
from agents import Agent
from app.memory import Memory
from openai import AsyncOpenAI
from app.config import env
from agents import set_default_openai_client, set_default_openai_api, set_tracing_disabled
from app.tool import caption, ocr, localize, qa, time

def build_agent():
    client = AsyncOpenAI(base_url=env.end_lang,api_key=env.api_key)
    set_default_openai_client(client=client, use_for_tracing=False)
    set_default_openai_api("chat_completions")
    set_tracing_disabled(disabled=True)
    chat_agent = Agent[Memory](  
        name="Assistant",
        tools=[caption, ocr, qa, time, localize],
        model=env.model_agent,
        instructions='As a helpful assistant, your functions include answering questions, Optical Character Recognition (OCR), image caption generation, and object localization within images. Additionally, you can set reminders based on user requests. A key requirement for setting reminders is to phrase the reminder check as a question that verifies the condition. For instance, if the user asks, "Remind me when I wear glasses," you should formulate the reminder check as: "Do I wear glasses?"',
    )

    return chat_agent

