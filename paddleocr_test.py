from paddleocr import PaddleOCR

img_path = "book.png"

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv6_small_det",
    text_recognition_model_name="PP-OCRv6_small_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,  
    engine="transformers",
    device="gpu:0",
)

output = ocr.predict(input=img_path)

for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")

    # 取出辨識文字
    data = res.json["res"]
    texts = data["rec_texts"]
    scores = data["rec_scores"]

    print("\n===== OCR 辨識結果 =====")
    for text, score in zip(texts, scores):
        print(f"{text}\t{score:.4f}")