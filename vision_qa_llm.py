from transformers import AutoProcessor, AutoModelForMultimodalLM
from PIL import Image
import torch, json, re

MODEL_NAME = "google/gemma-4-E2B-it"
DEFAULT_IMAGE = "book.png"

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_NAME,
    device_map="cuda",
)

SYSTEM_PROMPT = """你是書籍辨識助手。
請觀察圖片，列出你看到的書名。
只輸出 JSON，不要輸出其他文字。

格式：
{
  "book_name": "..." 
}"""


def parse_json_output(raw: str) -> dict:
    cleaned = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"answer": cleaned}


def vision_book_llm_infer(image_path: str = DEFAULT_IMAGE) -> dict:
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "請列出圖片中看到的書名。"},
            ],
        },
    ]

    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=text_prompt,
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    raw_text = processor.decode(
        outputs[0][input_len:],
        skip_special_tokens=True,
    )

    print(f"[VL Raw Output] {raw_text}")

    result = parse_json_output(raw_text)
    print(f"[Final Answer] {result}")
    return result


# vision_book_llm_infer()