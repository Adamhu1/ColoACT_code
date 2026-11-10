import cv2
import os
import numpy as np
from tqdm import tqdm

# === 1. 閰嶇疆鍖哄煙 ===
VIDEO_PATHS = {
    0: "data/images/",
    1: "data/videos/raw_polyp.mp4",
    2: "",
    3: "data/images/skill3/skill3_20251215_105144.avi"
}

# 杈撳嚭鐩綍 (浼氳嚜鍔ㄥ垱寤?
OUTPUT_ROOT = "outputs/vlm_dataset/test_rgb_crop13"
TARGET_COUNT = 30

# 鏄惁娑傞粦宸︿笂瑙掔殑鏂囧瓧鏍囩锛?寤鸿 True锛岄槻姝㈡暟鎹硠闇?
MASK_TEXT = True
MASK_W, MASK_H = 185, 30  # 閬尅鍖哄煙澶у皬锛屾牴鎹綘鐨勬埅鍥句及绠楃殑

def extract_frames(skill_id, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"鉂?鏃犳硶鎵撳紑: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < TARGET_COUNT:
        print(f"鈿狅笍 瑙嗛澶煭锛屽皢鎻愬彇鎵€鏈夊抚")
        interval = 1
    else:
        interval = total_frames // TARGET_COUNT

    save_dir = os.path.join(OUTPUT_ROOT, f"skill_{skill_id}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    count = 0
    saved_count = 0
    pbar = tqdm(total=TARGET_COUNT, desc=f"Skill {skill_id}")

    while cap.isOpened() and saved_count < TARGET_COUNT:
        ret, frame = cap.read()
        if not ret:
            break

        if count % interval == 0:
            # === 鏍稿績淇敼鍖哄煙 ===
            
            # 1. 鑾峰彇鍘熷浘灏哄 (Height, Width)
            h, w = frame.shape[:2]
            
            # 2. 瑁佸壀锛氬彧鍙栧乏涓婅 1/4 (RGB鍖哄煙)
            # 涔熷氨鏄珮搴︾殑鍓嶄竴鍗婏紝瀹藉害鐨勫墠涓€鍗?
            rgb_crop = frame[0 : h//2, 0 : w//2]
            
            # 3. (鍙€? 娑傞粦宸︿笂瑙掓枃瀛?"REC | Skill: 2"
            # 鏃㈢劧宸茬粡鍒囧嚭鏉ヤ簡锛岀幇鍦ㄦ槸鏈€濂界殑娑傞粦鏃舵満锛屼竴姝ュ埌浣?
            if MASK_TEXT:
                # 鍦ㄨ鍓悗鐨勫皬鍥句笂鐢婚粦妗?
                cv2.rectangle(rgb_crop, (0, 0), (MASK_W, MASK_H), (0, 0, 0), -1)

            # === 淇敼缁撴潫 ===

            # 4. 淇濆瓨鍥剧墖
            save_name = f"skill_{skill_id}_{saved_count:04d}.jpg"
            save_path = os.path.join(save_dir, save_name)
            cv2.imwrite(save_path, rgb_crop)
            
            saved_count += 1
            pbar.update(1)
        
        count += 1

    cap.release()
    pbar.close()

def main():
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    print("馃殌 寮€濮嬫彁鍙?RGB 鏁版嵁闆?(鑷姩瑁佸壀宸︿笂瑙?1/4)...")
    for skill_id, video_path in VIDEO_PATHS.items():
        extract_frames(skill_id, video_path)
    print(f"\n鉁?鍏ㄩ儴瀹屾垚锛佹暟鎹泦宸蹭繚瀛樿嚦: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()


