import cv2
import numpy as np

# 1. Load the specified material images
takeshi_img = cv2.imread("Takeshi_Exp.png", cv2.IMREAD_UNCHANGED)  # White areas (Takeshi with blonde hair)
normal_img = cv2.imread("Normal_Exp.png", cv2.IMREAD_UNCHANGED)    # Black areas (Normal bald variant)

if takeshi_img.shape[2] == 4:
    takeshi_img = takeshi_img[:, :, :3]
if normal_img.shape[2] == 4:
    normal_img = normal_img[:, :, :3]

# 2. Open the Bad Apple video
cap = cv2.VideoCapture("badApple.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # 480
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 360

# 3. Set a refined grid density (32 x 24, each cell is 15x15 pixels)
COLS = 32  
ROWS = 24  

cell_w = width // COLS   # 480 / 32 = 15 pixels
cell_h = height // ROWS  # 360 / 24 = 15 pixels

out_w = cell_w * COLS    # 480
out_h = cell_h * ROWS    # 360

# Create full-screen tiled background A (Takeshi for white areas) and B (Normal for black areas)
takeshi_tile = cv2.resize(takeshi_img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
normal_tile = cv2.resize(normal_img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)

bg_takeshi = np.zeros((out_h, out_w, 3), dtype=np.uint8)
bg_normal = np.zeros((out_h, out_w, 3), dtype=np.uint8)

for y in range(0, out_h, cell_h):
    for x in range(0, out_w, cell_w):
        bg_takeshi[y:y+cell_h, x:x+cell_w] = takeshi_tile
        bg_normal[y:y+cell_h, x:x+cell_w] = normal_tile

# Configure video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_bad_aooni_matte.mp4', fourcc, fps, (out_w, out_h))

print(f"Grid setup: {COLS}x{ROWS} (Refined Silhouette Edition) | Starting mask composition...")

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale and normalize mask to 0.0 ~ 1.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = gray.astype(np.float32) / 255.0
    mask = np.expand_dims(mask, axis=2)

    # White mask areas (mask=1) show Takeshi_Exp.png (Blonde)
    # Black mask areas (mask=0) show Normal_Exp.png (Bald)
    output_frame = (bg_takeshi * mask + bg_normal * (1.0 - mask)).astype(np.uint8)

    out.write(output_frame)
    frame_count += 1
    if frame_count % 300 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()
print("Complete! Output saved as output_bad_aooni_matte.mp4")