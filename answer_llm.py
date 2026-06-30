from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from model_load_4b_4bit import chat_model
from gemma_parser import strip_gemma

# 定義關鍵字提取的 Prompt
keyword_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是書名搜尋關鍵字提取助手。
請從使用者輸入中找出最重要的書籍搜尋關鍵字。

只輸出 JSON，格式如下：
{{
  "response": "簡短回覆okay",
  "key": "關鍵字"
}}

範例：
使用者：我要寵物那本書
輸出：
{{
  "response": "好的，我可以幫您找到關於寵物的那本書。",
  "key": "寵物"
}}

key 不要包含「我要」、「那本書」、「書」等無意義詞。"""),
    ("human", "{user_text}"),
])

json_parser = JsonOutputParser()

keyword_chain = keyword_prompt | chat_model | strip_gemma | json_parser

def answer_llm_infer(user_text: str) -> dict:

    result = keyword_chain.invoke({
        "user_text": user_text
    })

    return {
        "response": result.get("response", ""),
        "key": result.get("key", "")
    }