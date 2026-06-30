import json
import re
import os

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from model_load_4b_4bit import chat_model


# =========================
# JSON Output Parser
# =========================

json_parser = JsonOutputParser()


def extract_first_json(text: str) -> dict:
    """Fallback JSON extractor (used when JsonOutputParser isn't in the chain)."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"No JSON found in model output:\n{text}")
    return json.loads(match.group(0))


# =========================
# LangChain LLM caller
# =========================

def run_llm(messages: list[dict], max_new_tokens: int = 256) -> str:
    """
    Drop-in replacement for the old run_llm().
    messages: list of {"role": "system"/"user"/"assistant", "content": "..."}
    Returns the raw string output.
    """
    lc_messages = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))

    # Temporarily patch max_new_tokens on the underlying pipeline
    chat_model.llm.pipeline._forward_params = {"max_new_tokens": max_new_tokens}

    response = chat_model.invoke(lc_messages)
    return response.content.strip()


# =========================
# Markdown knowledge loader
# =========================

def parse_front_matter(md_text: str) -> dict:
    result = {}
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?", md_text, flags=re.DOTALL)
    if not match:
        return result
    for line in match.group(1).splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def remove_front_matter(md_text: str) -> str:
    return re.sub(r"^---\s*\n.*?\n---\s*\n?", "", md_text, flags=re.DOTALL)


def load_markdown_items(base_dir: str) -> list[dict]:
    items = []
    for filename in os.listdir(base_dir):
        if not filename.endswith(".md"):
            continue
        path = os.path.join(base_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        meta = parse_front_matter(text)
        content = remove_front_matter(text).strip()
        items.append({
            "filename": filename,
            "name": meta.get("name", filename[:-3]),
            "description": meta.get("description", ""),
            "content": content,
            "path": path,
        })
    return items


def get_item_by_name(items: list[dict], name: str) -> dict:
    for item in items:
        if item["name"] == name:
            return item
    raise ValueError(f"Item not found: {name}")


def build_skill_catalog(skills: list[dict]) -> str:
    return "\n".join(
        f'- "{s["name"]}": {s["description"]}'
        for s in skills
    )


# =========================
# Chat History (LangChain)
# =========================

MAX_HISTORY = 4

# In-memory store keyed by session_id
_history_store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _history_store:
        _history_store[session_id] = InMemoryChatMessageHistory()
    return _history_store[session_id]


def build_history_prompt(user_input: str, chat_history: list[dict]) -> str:
    """Legacy helper kept for compatibility."""
    if chat_history:
        recent = chat_history[-(MAX_HISTORY * 2):]
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in recent])
        return f"對話紀錄：\n{history_str}\n\n現在使用者：{user_input}"
    return user_input


def update_history(chat_history: list[dict], user_input: str, response_text: str):
    """Legacy helper kept for compatibility."""
    chat_history.append({"role": "使用者", "content": user_input})
    chat_history.append({"role": "你回覆", "content": response_text})
