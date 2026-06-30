# Multi-Modal Book Search System

本專案是一套多模態書籍搜尋系統，目標是讓使用者可以用自然語言描述想找的書，系統再從單張書架影像中找出最可能的目標書本，並輸出該書的位置。

系統整合了大型語言模型、YOLO 書本偵測、PP-OCRv6 文字辨識、sentence embedding 相似度比對，以及 VLM 視覺語言模型驗證。

![Multi-Modal AI Pipeline](Gemini_Generated_Image_e3kpgze3kpgze3kp.png)

## System Flow

整體流程如下：

```text
使用者輸入搜尋指令
  ↓
Gemma4 LLM 提取書名關鍵字
  ↓
YOLO 偵測並分割書本區域
  ↓
PP-OCRv6 辨識每一本書上的文字
  ↓
Sentence embedding 計算 OCR 文字與關鍵字相似度
  ↓
取 OCR 排名前 3 名候選書本
  ↓
VLM 依序檢查候選 crop
  ↓
若 VLM 相似度超過門檻，立即輸出結果
  ↓
若前 3 名皆未達門檻，輸出 VLM 分數最高者
```

## Modules

| File | Function |
|---|---|
| `answer_llm.py` | 使用 Gemma4 LLM 從使用者輸入中提取搜尋關鍵字 |
| `sg_text_v2.py` | 執行 YOLO 書本偵測與 PP-OCRv6 文字辨識 |
| `text_similarity.py` | 使用 sentence embedding 計算文字相似度 |
| `vision_qa_llm.py` | 使用 VLM 讀取候選書本 crop 中的書名 |
| `main_book_search_config.py` | 整合完整搜尋流程 |

## Keyword Extraction

使用者可以輸入自然語言，例如：

```text
我要寵物那本書
```

LLM 會將句子轉換成搜尋關鍵字：

```json
{
  "response": "好的，我可以幫您找到關於寵物的那本書。",
  "key": "寵物"
}
```

後續系統會使用 `key` 作為搜尋目標。

## YOLO and OCR

系統會先使用 YOLO segmentation model 偵測書本，取得每一本書的：

```text
bbox
mask
crop image
```

接著使用 PP-OCRv6 對每一本書的 crop 進行文字辨識，得到每本書的 OCR 結果。

## Similarity Matching

OCR 文字不一定完全正確，因此系統不只做字串包含比對，而是使用 sentence embedding 進行相似度計算。

比對方式會將 OCR 文字切成多個文字片段，再與搜尋關鍵字計算語意相似度，並加入字元重疊加權分數。

例如：

```text
目標關鍵字：寵物
OCR 文字：我的第一本寵物照護百科
```

系統會找出最接近 `寵物` 的文字片段，並依照分數排序候選書本。

## VLM Verification

OCR 排名前 3 名會再交給 VLM 進行視覺確認。

VLM 不會直接看整張書架圖，而是只看單一本書的 crop，避免受到旁邊書本干擾。

檢查邏輯如下：

```text
1. 從 OCR 第一名開始交給 VLM 辨識
2. 將 VLM 輸出的書名與搜尋關鍵字計算相似度
3. 如果分數高於固定門檻，立即停止並輸出該書
4. 如果沒有通過門檻，繼續檢查下一名
5. 若前三名都未通過門檻，輸出 VLM 分數最高的候選
```

這樣可以減少不必要的 VLM 推理時間，同時保留二次確認能力。

## Output

系統最後會輸出：

```json
{
  "user_query": "我要寵物那本書",
  "key": "寵物",
  "found": true,
  "bbox": [x1, y1, x2, y2],
  "ocr_text": "...",
  "ocr_score": 0.0,
  "vlm_text": "...",
  "vlm_score": 0.0,
  "selection_mode": "vlm_threshold_accept",
  "result_image": "out_single_image_result/best_image.png"
}
```

其中 `result_image` 會在原圖上標示出最終選到的書本位置。

## Configuration

主要設定集中在 `main_book_search_config.py` 上方：

```python
USER_QUERY = "我要寵物那本書"
IMAGE_PATH = "book.png"
YOLO_MODEL_PATH = "book_seg.pt"
TOP_K_FOR_VLM = 3
VLM_ACCEPT_THRESHOLD = 2.5
OUT_DIR = "out_single_image_result"
```

若要更換圖片、搜尋文字或 VLM 門檻，只需要修改這些設定。

## Run

執行完整流程：

```bash
python main_book_search_config.py
```

執行後會產生：

```text
out_single_image_result/best_image.png
out_single_image_result/final_result.json
```

## Summary

本系統並不完全依賴單一模型，而是採用多階段驗證流程：

```text
LLM 理解使用者需求
YOLO 找出候選書本
OCR 讀取書名文字
Embedding 進行文字相似度排序
VLM 對高分候選進行視覺確認
```

透過這種方式，系統可以降低單純 OCR 錯字造成的誤判，也避免直接使用 VLM 處理整張圖片時產生不穩定或幻覺問題。
