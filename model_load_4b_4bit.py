import os
from langchain_community.chat_models import ChatLlamaCpp
# from langchain_core.caches import InMemoryCache
# from langchain_core.globals import set_llm_cache

# set_llm_cache(InMemoryCache())

pwd = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(pwd, "gemma-4-E2B-it-Q4_K_M.gguf")

if not os.path.exists(model_path):
    print(f"no model：{model_path}")
else:

    chat_model = ChatLlamaCpp(
        model_path=model_path,
        n_gpu_layers=-1, 
        n_ctx=1200,         # 上下文長度
        f16_kv=True,
        verbose=True,
        temperature=0,      
        max_tokens=300,
        # cache=True,
    )

    print("Chat model (GGUF) 載入成功！")

