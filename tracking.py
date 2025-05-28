from facenet_pytorch import MTCNN
import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw
import time

# デバイスの設定
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
mtcnn = MTCNN(keep_all=True, device=device)

# ビデオファイルの読み込み
video = mmcv.VideoReader('./faces/video.mp4')
frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
print(f'Total frames: {len(frames)}')

# 顔検出と追跡処理
frames_tracked = []
for i, frame in enumerate(frames):
    print('\rTracking frame: {}/{}'.format(i + 1, len(frames)), end='')
    
    # 顔の検出
    boxes, _ = mtcnn.detect(frame)
    
    # 顔の周りに枠を描画
    frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)
    if boxes is not None:  # Noneチェックを追加
        for box in boxes:
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
    
    # フレームリストに追加
    frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
print('\nDone')

# OpenCVを使用してリアルタイムでフレームを表示
print('Displaying frames (press q to exit)...')
for frame in frames_tracked:
    # PIL ImageをOpenCV形式に変換
    cv_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    cv2.imshow('Tracked Faces', cv_frame)
    
    # 'q'キーで終了、25fpsのレートでフレーム間を待機
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# 処理結果をビデオファイルとして保存
print('Saving video...')
dim = frames_tracked[0].size
fourcc = cv2.VideoWriter_fourcc(*'FMP4')    
video_tracked = cv2.VideoWriter('video_tracked.mp4', fourcc, 25.0, dim)
for frame in frames_tracked:
    video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
video_tracked.release()
print('Video saved as video_tracked.mp4')