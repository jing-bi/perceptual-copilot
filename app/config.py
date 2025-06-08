import os
import logging
from openai import OpenAI




try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class Envs:
    def __init__(self):
        self.hf_token = self._get_env("HF_TOKEN")
        self.api_key = self._get_env("API_KEY")
        self.end_task = self._get_env("END_TASK")
        self.end_lang = os.getenv("END_LANG")
        self.model_agent = os.getenv("MODEL_AGENT")
        self.model_mllm = os.getenv("MODEL_MLLM")
        self.model_loc = os.getenv("MODEL_LOC")
        self.client = OpenAI(base_url=self.end_lang, api_key=self.api_key)
        self.debug = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
    
    def _get_env(self, key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} is not set")
        return value

env = Envs()

logger = logging.getLogger('copilot')
logger.setLevel(logging.DEBUG if env.debug else logging.INFO)
logger.addHandler(logging.StreamHandler())