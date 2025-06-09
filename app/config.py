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
        
        # Only initialize OpenAI client if we have the required env vars
        if self.end_lang and self.api_key:
            self.client = OpenAI(base_url=self.end_lang, api_key=self.api_key)
        else:
            self.client = None
            print("WARNING: OpenAI client not initialized due to missing environment variables")
            
        self.debug = os.getenv("DEBUG", "1").lower() in ("true", "1", "yes")
    
    def _get_env(self, key: str) -> str:
        value = os.getenv(key)
        if not value:
            # For HF Spaces, provide fallback or warning
            if key == "HF_TOKEN":
                # HF Spaces automatically provides this
                value = os.getenv("HF_TOKEN", "")
            else:
                print(f"WARNING: Environment variable {key} is not set")
                # Return empty string instead of raising error for deployment
                return ""
        return value

env = Envs()

logger = logging.getLogger('copilot')
logger.setLevel(logging.DEBUG if env.debug else logging.INFO)
logger.addHandler(logging.StreamHandler())