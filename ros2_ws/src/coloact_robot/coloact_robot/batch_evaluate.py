import os
import requests
import base64
import cv2
import pandas as pd
from tqdm import tqdm  # 杩涘害鏉″簱锛宲ip install tqdm
import time

# === 閰嶇疆 ===
DATASET_ROOT = "outputs/vlm_dataset/test"  # 浣犵殑鏁版嵁璺緞
SERVER_URL = "http://127.0.0.1:8000/plan"       # Docker VLM 鍦板潃
OUTPUT_FILE = "vlm_evaluation_results.csv"

# 浣犵殑 Prompt (蹇呴』鍜?vlm_planner_node.py 閲岀殑涓€妯′竴鏍凤紝淇濊瘉娴嬭瘯涓€鑷存€?
SYSTEM_PROMPT = """
You are a colonoscopy robot navigation assistant operating in a pig colon. 
Analyze the image and output the correct Skill ID (0, 1, 2, or 3).

Decision Logic Chain (Step-by-Step):
1. [HIGHEST PRIORITY] TARGET DETECTION:
   - Is there a RED MARKER or RED POLYP visible? -> YES: Output 1.
2. [HIGH PRIORITY] EMERGENCY CHECK:
   - Is the view completely blocked or blurry? -> YES: Output 3.
3. [NORMAL PRIORITY] NAVIGATION GEOMETRY:
   - Is the lumen hidden or S-turn? -> YES: Output 2.
   - Is the lumen centered? -> YES: Output 0.

Constraint: Reply ONLY with the single digit (e.g., 0).
"""

def get_vlm_prediction(image_path):
    try:
        # 璇诲彇骞剁紪鐮佸浘鐗?
        img = cv2.imread(image_path)
        if img is None: return -1
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # 鍙戦€佽姹?
        payload = {"image_base64": img_base64, "prompt": SYSTEM_PROMPT}
        response = requests.post(SERVER_URL, json=payload, timeout=5)
        
        if response.status_code == 200:
            res_text = response.json().get("result", "")
            # 绠€鍗曠殑瑙ｆ瀽閫昏緫
            import re
            match = re.search(r'[0-3]', res_text)
            return int(match.group()) if match else -1
    except Exception as e:
        print(f"Error: {e}")
    return -1

def main():
    results = []
    
    # 閬嶅巻 4 涓?Skill 鏂囦欢澶?
    for true_label in [0, 1, 2, 3]:
        folder_path = os.path.join(DATASET_ROOT, f"skill_{true_label}")
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found {folder_path}")
            continue
            
        images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
        print(f"Processing Skill {true_label}: {len(images)} images...")

        for img_name in tqdm(images):
            img_path = os.path.join(folder_path, img_name)
            
            # 璋冪敤 VLM
            pred_label = get_vlm_prediction(img_path)
            
            results.append({
                "image_name": img_name,
                "true_label": true_label,
                "pred_label": pred_label
            })
            
            # 绋嶅井寤舵椂涓€涓嬶紝闃叉鎶婃樉鍗¤窇宕?
            # time.sleep(0.05)

    # 淇濆瓨缁撴灉
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"鉁?璇勪及瀹屾垚锛佺粨鏋滃凡淇濆瓨鑷?{OUTPUT_FILE}")

if __name__ == "__main__":
    main()


