from dotenv import load_dotenv
from config import SRC
import dspy
import os

def configure_lm(model_str: str) -> None:
    load_dotenv(SRC / ".env")
    api_key = os.getenv("OPENROUTER_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_KEY is not set in the environment")
    dspy.configure(lm=dspy.LM(
        model=model_str,
        api_base="https://openrouter.ai/api/v1",
        api_key=api_key,
    ))