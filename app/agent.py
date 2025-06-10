
from agents import Agent
from app.memory import Memory
from openai import AsyncOpenAI
from app.config import env
from agents import set_default_openai_client, set_default_openai_api, set_tracing_disabled
from app.tool import caption, ocr, localize, qa, time, video_caption, video_qa

def build_agent():
    client = AsyncOpenAI(base_url=env.end_lang,api_key=env.api_key)
    set_default_openai_client(client=client, use_for_tracing=False)
    set_default_openai_api("chat_completions")
    set_tracing_disabled(disabled=True)
    chat_agent = Agent[Memory](
        name="Assistant",
        tools=[caption, ocr, qa, time, localize, video_caption, video_qa],
        model=env.model_agent,
        instructions=(
            "As a helpful assistant, your functions include answering questions about images, "
            "Optical Character Recognition (OCR), image caption generation, object localization "
            "within images, and video caption generation and Q&A. For video-related tools, you "
            "will need to determine the appropriate time window to analyze from the past."
        ),
    )

    return chat_agent

