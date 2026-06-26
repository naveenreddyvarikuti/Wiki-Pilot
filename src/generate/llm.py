from pathlib import Path
from langchain_community.llms import LlamaCpp
from langchain_core.documents import Document
from src.generate.prompt_loader import load_prompt, format_prompt

MODEL_PATH = str(Path("models/qwen2.5-7b-instruct-q3_k_m.gguf"))
DEFAULT_PROMPT_VERSION = "v2"


def build_llm() -> LlamaCpp:
    return LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=8,
        temperature=0.1,
        max_tokens=512,
        verbose=False,
    )


def generate_answer(
    query: str,
    sources: list[Document],
    llm: LlamaCpp,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> str:
    prompt_config = load_prompt(prompt_version)
    prompt = format_prompt(prompt_config, query, sources)
    return llm.invoke(prompt)
