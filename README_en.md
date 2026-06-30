# Multi-Modal Book Search System

This project is a multi-modal book search system. Users can describe the book they want to find using natural language, and the system will locate the most likely target book from a single bookshelf image and output its position in the image.

The system integrates a large language model, YOLO-based book detection, PP-OCRv6 text recognition, sentence embedding similarity matching, and VLM-based visual verification.

![Multi-Modal AI Pipeline](Gemini_Generated_Image_e3kpgze3kpgze3kp.png)

## System Flow

The overall pipeline is as follows:

```text
User search query
  ↓
Gemma4 LLM extracts the book keyword
  ↓
YOLO detects and segments book regions
  ↓
PP-OCRv6 recognizes text from each detected book
  ↓
Sentence embedding computes similarity between OCR text and keyword
  ↓
Select top 3 OCR candidates
  ↓
VLM sequentially verifies each candidate crop
  ↓
If VLM similarity exceeds the threshold, output the result immediately
  ↓
If none of the top 3 candidates passes the threshold, output the one with the highest VLM score
```

## Modules

| File | Function |
|---|---|
| `answer_llm.py` | Uses Gemma4 LLM to extract search keywords from the user query |
| `sg_text_v2.py` | Runs YOLO book detection and PP-OCRv6 text recognition |
| `text_similarity.py` | Computes text similarity using sentence embeddings |
| `vision_qa_llm.py` | Uses a VLM to read the book title from each candidate crop |
| `main.py` | Integrates the full search pipeline |
| `book_seg.pt` | YOLO book segmentation model weights |
| `model_load_4b_4bit.py` | Loads the LLM model |

## Keyword Extraction

Users can enter a natural language query, for example:

```text
I want the pet book
```

The LLM converts the query into a search keyword:

```json
{
  "response": "Okay, I can help you find the book about pets.",
  "key": "pet"
}
```

The system then uses `key` as the search target.

## YOLO and OCR

The system first uses a YOLO segmentation model to detect books and obtain the following information for each book:

```text
bbox
mask
crop image
```

Then, PP-OCRv6 is applied to each cropped book image to recognize text.

Since OCR output may contain wrong characters, missing characters, or extra text, the system does not rely on exact string matching. Instead, it uses similarity matching to rank candidate books.

## Similarity Matching

The core function for similarity scoring is:

```python
get_best_window_similarity(source_text, target_text)
```

This function belongs to the `TextSimilarityMatcher` class.

It compares:

| Input | Meaning |
|---|---|
| `source_text` | Text recognized by OCR or VLM |
| `target_text` | The target keyword extracted from the user query |

To improve matching accuracy, the system uses a sliding window strategy and combines semantic similarity with character overlap.

### 1. Text Normalization

The system first removes spaces, line breaks, and tabs to avoid formatting differences affecting the similarity score.

### 2. Sliding Window

OCR text is usually longer than the search keyword, so the system splits `source_text` into multiple short segments.

```text
window_size = len(target_text) + padding
```

Each segment is then compared with `target_text`.

### 3. Semantic Score

The system uses `SentenceTransformer` to convert the target text and each segment into embeddings, then computes cosine similarity:

```text
Semantic Score = (A · B) / (||A|| ||B||)
```

where `A` is the embedding of the target keyword and `B` is the embedding of the OCR/VLM segment.

### 4. Character Overlap Bonus

Semantic similarity alone may not be stable enough for book titles, especially when the title contains short words, proper nouns, or OCR errors. Therefore, the system also adds a character overlap bonus.

```text
Overlap Ratio = matched target characters / total unique target characters
Bonus = Overlap Ratio × Bonus Weight
```

Different weights are used in different stages:

| Stage | Bonus Weight |
|---|---|
| OCR similarity | `1.0` |
| VLM similarity | `3.0` |

The VLM stage uses a higher weight because the VLM only sees a single book crop, so character-level matches are usually more reliable.

### 5. Final Score

The final score for each segment is:

```text
Final Score = Semantic Score + (Overlap Ratio × Bonus Weight)
```

The system returns the highest score and the best-matching segment.

## VLM Verification

The top 3 OCR candidates are verified by the VLM.

The VLM does not process the entire bookshelf image. Instead, it only sees one cropped book at a time, which reduces interference from nearby books.

The verification logic is:

```text
1. Start from the top OCR candidate
2. Use the VLM to recognize the book title from the crop
3. Compute similarity between the VLM output and the target keyword
4. If the score exceeds the threshold, stop immediately and output this book
5. Otherwise, continue to the next candidate
6. If none of the top 3 candidates passes the threshold, output the one with the highest VLM score
```

## Output

The system outputs a result similar to:

```json
{
  "user_query": "I want the pet book",
  "key": "pet",
  "found": true,
  "bbox": [x1, y1, x2, y2],
  "ocr_text": "...",
  "ocr_score": 0.0,
  "ocr_match": "...",
  "vlm_text": "...",
  "vlm_score": 0.0,
  "vlm_match": "...",
  "selection_mode": "vlm_threshold_accept",
  "result_image": "out_single_image_result/best_image.png"
}
```

`result_image` marks the selected book location on the original image.

## Configuration

Main settings are defined at the top of `main.py`:

```python
USER_QUERY = "我要寵物那本書"
IMAGE_PATH = "book.png"
YOLO_MODEL_PATH = "book_seg.pt"
TOP_K_FOR_VLM = 3
VLM_ACCEPT_THRESHOLD = 2.5
OUT_DIR = "out_single_image_result"
```

You can modify these settings to change the input image, query, or VLM threshold.

## Environment Setup

Python 3.11 is recommended. Python 3.10 may also work, but the main tested environment uses Python 3.11.

### 1. Create a Conda Environment

```bash
conda create -n book_grasping python=3.11 -y
conda activate book_grasping

python -m pip install --upgrade pip setuptools wheel
```

### 2. Install PyTorch

Install PyTorch according to your CUDA version.

For example, CUDA 12.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Check your CUDA version with:

```bash
nvidia-smi
```

### 3. Install llama-cpp-python

Install the CUDA version of `llama-cpp-python`.

Reference:

https://pypi.org/project/llama-cpp-python/

Installation format:

```bash
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/<cuda-version>
```

For example, CUDA 12.5:

```bash
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125
```

Even if `nvidia-smi` shows CUDA 13.0, `cu125` is recommended for better compatibility.

### 4. Install PaddleOCR

```bash
pip install "paddleocr[all]"
```

### 5. Install YOLO and Image Processing Packages

```bash
pip install ultralytics
pip install opencv-python pillow numpy
```

### 6. Install LLM / VLM Packages

```bash
pip install transformers accelerate safetensors sentencepiece protobuf huggingface_hub
```

### 7. Install LangChain Packages

```bash
pip install langchain-core langchain-community
```

### 8. Install Sentence Embedding Package

```bash
pip install sentence-transformers
```

### 9. Download VLM, PaddleOCR, and Embedding Models

Some models require a Hugging Face account. Please create one first:

https://huggingface.co/

When running `main.py`, the system may ask you to log in to Hugging Face, and the required models will be downloaded automatically.

### 10. Download the LLM Model

Download the GGUF model from:

https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF

Use:

```text
gemma-4-E2B-it-Q4_K_M.gguf
```

Place the downloaded model file in the same directory as the other project files.

## Run

Run the full pipeline:

```bash
python main.py
```

The outputs will be saved to:

```text
out_single_image_result/best_image.png
out_single_image_result/final_result.json
```

## Notes

The current version supports single-image inference only. To connect it to a real-time camera, replace the image loading part with a camera frame input.

If you encounter environment setup issues, feel free to ask, since the environment can be difficult to build for users who are not familiar with CUDA and local LLM deployment.
