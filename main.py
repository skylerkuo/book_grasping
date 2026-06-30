import os
import cv2
import tempfile
import numpy as np

from answer_llm import answer_llm_infer
from sg_text_v2 import YOLOProcessor, OCRProcessor, process_books_pipeline
from text_similarity import TextSimilarityMatcher
from vision_qa_llm import vision_book_llm_infer

def main():
    # --- 1. 設定與模型初始化 ---
    TARGET_IMAGE = "book.png"  
    VLM_THRESHOLD = 2.5
    
    print("=" * 50)
    print("[*] 初始化各項模型中")
    yolo_proc = YOLOProcessor(model_path="book_seg.pt")
    ocr_proc = OCRProcessor(device="gpu:0")
    similarity_matcher = TextSimilarityMatcher()

    print("\n" + "=" * 50)
    print("[*] 輸入 'q', 'quit' 或 'exit' 來退出程式。")
    print("=" * 50)

    # --- 2. 互動式與即時推論迴圈 ---
    while True:
        try:
            # 取得使用者指令
            user_query = input("\n>>> 請輸入搜尋指令 (或輸入 q 退出): ").strip()
            
            if not user_query:
                continue
                
            if user_query.lower() in ['q', 'quit', 'exit']:
                print("[*] 結束程式。")
                break
            
            # --- [階段 A]: LLM 意圖解析 ---
            print(f"[*] 處理請求: '{user_query}'")
            llm_intent = answer_llm_infer(user_query)
            goal_text = llm_intent.get("key", "").strip()
            
            if not goal_text:
                print("[!] 無法從 LLM 提取出有效的搜尋關鍵字，請重新輸入。")
                continue
                
            print(f"[*] 鎖定目標關鍵字: '{goal_text}'")

            # --- [階段 B]: 讀取當下畫面、YOLO 偵測與 OCR (每次皆重新執行) ---
            if not os.path.exists(TARGET_IMAGE):
                print(f"[!] 找不到測試圖片: {TARGET_IMAGE}")
                continue
                
            print(f"\n[*] 正在對當前畫面執行 YOLO 切割與 OCR 辨識...")
            ocr_candidates = process_books_pipeline(TARGET_IMAGE, yolo_proc, ocr_proc)
            
            if not ocr_candidates:
                print("[!] 當前畫面中未偵測到任何書本，請調整鏡頭後再試。")
                continue

            full_bgr = cv2.imread(TARGET_IMAGE)

            # --- [階段 C]: 相似度比對與排序 ---
            scored_candidates = []
            for idx, item in enumerate(ocr_candidates):
                ocr_text = item.get("text", "")
                score, best_seg = similarity_matcher.get_best_window_similarity(
                    source_text=ocr_text, 
                    target_text=goal_text, 
                    bonus_weight=1
                )
                scored_candidates.append({
                    "idx": idx,
                    "info": item,
                    "ocr_text": ocr_text,
                    "ocr_score": score,
                    "best_seg": best_seg
                })

            # 取 OCR 相似度前 3 名
            scored_candidates.sort(key=lambda x: x["ocr_score"], reverse=True)
            top_3_candidates = scored_candidates[:3]
            
            print("\n[*] 當前畫面 OCR 篩選前 3 名:")
            for rank, cand in enumerate(top_3_candidates, start=1):
                print(f"    Rank {rank}: Score={cand['ocr_score']:.4f} | Text='{cand['ocr_text']}'")

            # --- [階段 D]: VLM 密集驗證 (加入提前終止邏輯) ---
            print(f"\n[*] 開始透過 VLM 驗證 Top 3 候選 (達標門檻: {VLM_THRESHOLD})...")
            
            best_final_result = None
            fallback_result = None
            highest_vlm_score = -1.0

            with tempfile.TemporaryDirectory() as temp_dir:
                for rank, cand in enumerate(top_3_candidates, start=1):
                    info = cand["info"]
                    x1, y1, x2, y2 = info["bbox"]
                    mask = info["mask"]
                    
                    # 使用 Mask 進行去背裁切
                    crop = full_bgr[y1:y2, x1:x2].copy()
                    mask_crop = mask[y1:y2, x1:x2]
                    crop[mask_crop == 0] = 0
                    
                    # 存下暫存圖供 VLM 讀取
                    temp_crop_path = os.path.join(temp_dir, f"candidate_{rank}.png")
                    cv2.imwrite(temp_crop_path, crop)
                    
                    print(f"    -> 正在分析 Rank {rank}...")
                    vlm_response = vision_book_llm_infer(temp_crop_path)
                    vlm_text = str(vlm_response.get("book_name", vlm_response.get("answer", "")))
                    
                    # 重新計算 VLM 的語意分數
                    vlm_score, vlm_best_seg = similarity_matcher.get_best_window_similarity(
                        source_text=vlm_text,
                        target_text=goal_text,
                        bonus_weight=3.0
                    )
                    
                    print(f"       VLM 辨識結果: '{vlm_text}' | Score: {vlm_score:.4f}")
                    
                    # 將當前結果包裝起來
                    current_result = {
                        "rank_in_ocr": rank,
                        "bbox": info["bbox"],
                        "mask": mask,  
                        "vlm_text": vlm_text,
                        "vlm_score": vlm_score,
                        "ocr_text": cand["ocr_text"],
                        "ocr_score": cand["ocr_score"]
                    }

                    # 更新最高分紀錄，作為後續的備案 (fallback)
                    if vlm_score > highest_vlm_score:
                        highest_vlm_score = vlm_score
                        fallback_result = current_result

                    # 【新增邏輯】分數大於等於門檻，直接鎖定並中斷迴圈
                    if vlm_score >= VLM_THRESHOLD:
                        print(f"    [!] 分數達標 (>= {VLM_THRESHOLD})，提前鎖定目標，跳過後續候選驗證。")
                        best_final_result = current_result
                        break
                        
            # 【新增邏輯】若跑完迴圈都沒有任何一個超過門檻，則輸出最高分的那個
            if best_final_result is None and fallback_result is not None:
                print(f"\n    [!] 均未達門檻，退而求其次選擇最高分之結果。")
                best_final_result = fallback_result

            # --- [階段 E]: 輸出最終結果與視覺化 ---
            if best_final_result:
                print("\n" + "=" * 50)
                print("[V] 畫面分析完畢，最終鎖定目標:")
                print(f"    目標關鍵字 : {goal_text}")
                print(f"    VLM Text   : {best_final_result['vlm_text']}")
                print(f"    VLM Score  : {best_final_result['vlm_score']:.4f}")
                print(f"    選中來源   : OCR Rank {best_final_result['rank_in_ocr']}")
                
                x1, y1, x2, y2 = best_final_result["bbox"]
                best_mask = best_final_result["mask"]

                # 畫 Mask 半透明圖層
                color_mask = np.zeros_like(full_bgr)
                color_mask[best_mask > 0] = [0, 255, 0]
                cv2.addWeighted(color_mask, 0.3, full_bgr, 1.0, 0, full_bgr)

                # 畫 Mask 輪廓與 Bounding Box
                contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(full_bgr, contours, -1, (0, 255, 0), 2)
                cv2.rectangle(full_bgr, (x1, y1), (x2, y2), (0, 0, 255), 4)
                
                # 儲存結果
                out_path = "camera_frame_result.png"
                cv2.imwrite(out_path, full_bgr)
                print(f"\n[!] 已將當前畫面標註結果儲存至: {out_path}")
            else:
                print("\n[X] VLM 驗證失敗，當前畫面未找到符合目標。")
                
        except Exception as e:
            print(f"\n[!] 執行過程中發生錯誤: {str(e)}")
            print("[*] 系統重置中，請重新輸入下一道指令...")

if __name__ == "__main__":
    main()