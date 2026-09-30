import os

from dotenv import load_dotenv

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))

load_dotenv(dotenv_path=env_path)


class Settings:

    def __init__(self):


        self.groq_api_key = os.getenv(
            "GROQ_API_KEY",
        )

        self.llm_model = os.getenv(
            "LLM_MODEL",
            "openai/gpt-oss-120b"
        )

        self.chroma_dir = os.getenv(
            "CHROMA_DIR",
            "./chroma_db"
        )

        self.validate()

    def validate(self):

        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is missing."
            )


settings = Settings()
