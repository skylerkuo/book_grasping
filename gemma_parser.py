"""
gemma_parser.py
---------------
Strips Gemma 4 chat-template tokens from raw model output before JSON parsing.

Gemma 4 (E4B-it via HuggingFacePipeline) echoes the full prompt back together
with the completion, wrapped in tokens like:
    <bos><|turn>system ... <|turn>model\n{ ... }\n<turn|>

LangChain's JsonOutputParser receives this full string and raises:
    JSONDecodeError: Expecting value: line 1 column 1

Solution: insert `strip_gemma | json_parser` instead of just `json_parser`.
"""

import re
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


def _strip(message) -> AIMessage:
    text = message.content if hasattr(message, "content") else str(message)

    # Gemma 4 template: <|turn>model\n...\n<turn|>
    m = re.search(r"<\|turn>model\s*(.*?)\s*<turn\|>", text, re.DOTALL)
    if m:
        return AIMessage(content=m.group(1).strip())

    # Gemma 3 / older template: <start_of_turn>model\n...<end_of_turn>
    m = re.search(r"<start_of_turn>model\s*(.*?)(?:<end_of_turn>|$)", text, re.DOTALL)
    if m:
        return AIMessage(content=m.group(1).strip())

    # Fallback: remove known special tokens and return as-is
    cleaned = re.sub(r"<bos>|<eos>|<\|turn>[a-z]*|<turn\|>", "", text).strip()
    return AIMessage(content=cleaned)


# Drop-in RunnableLambda — use between chat_model and json_parser
strip_gemma = RunnableLambda(_strip)