import cv2
import os
from pathlib import Path

def extract_frames(video_path, output_folder, frame_interval=1, image_format='jpg'):
    """
    動画からフレームを抽出して指定フォルダに保存
    
    Args:
        video_path (str): 動画ファイルのパス
        output_folder (str): 保存先フォルダのパス
        frame_interval (int): フレーム抽出間隔（1=全フレーム、30=30フレームごと）
        image_format (str): 保存画像の形式（'jpg', 'png', 'bmp'など）
    """
    
    # 出力フォルダを作成
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # 動画ファイルを開く
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"エラー: 動画ファイル '{video_path}' を開けませんでした")
        return
    
    # 動画の情報を取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"動画情報:")
    print(f"  総フレーム数: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  再生時間: {total_frames/fps:.2f}秒")
    print(f"  抽出間隔: {frame_interval}フレームごと")
    print(f"  保存形式: {image_format}")
    print(f"  保存先: {output_folder}")
    print()
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 指定間隔でフレームを保存
        if frame_count % frame_interval == 0:
            # ファイル名を生成（ゼロパディング）
            filename = f"frame_{frame_count:06d}.{image_format}"
            filepath = os.path.join(output_folder, filename)
            
            # フレームを保存
            cv2.imwrite(filepath, frame)
            saved_count += 1
            
            if saved_count % 100 == 0:
                print(f"保存済み: {saved_count}フレーム")
        
        frame_count += 1
    
    cap.release()
    
    print(f"\n完了!")
    print(f"総処理フレーム数: {frame_count}")
    print(f"保存フレーム数: {saved_count}")

def extract_frames_by_time(video_path, output_folder, time_interval=1.0, image_format='jpg'):
    """
    動画から指定時間間隔でフレームを抽出
    
    Args:
        video_path (str): 動画ファイルのパス
        output_folder (str): 保存先フォルダのパス
        time_interval (float): 時間間隔（秒）
        image_format (str): 保存画像の形式
    """
    
    # 出力フォルダを作成
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"エラー: 動画ファイル '{video_path}' を開けませんでした")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"動画情報:")
    print(f"  再生時間: {duration:.2f}秒")
    print(f"  FPS: {fps:.2f}")
    print(f"  時間間隔: {time_interval}秒ごと")
    print(f"  保存先: {output_folder}")
    print()
    
    saved_count = 0
    current_time = 0.0
    
    while current_time < duration:
        # 指定時間のフレームに移動
        frame_number = int(current_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # ファイル名を生成（時間ベース）
        filename = f"frame_{current_time:06.2f}s.{image_format}"
        filepath = os.path.join(output_folder, filename)
        
        cv2.imwrite(filepath, frame)
        saved_count += 1
        
        print(f"保存: {filename} ({current_time:.2f}秒)")
        
        current_time += time_interval
    
    cap.release()
    
    print(f"\n完了! 保存フレーム数: {saved_count}")

# 使用例
if __name__ == "__main__":
    # 設定
    VIDEO_PATH = "./faces/video.mp4"  # 入力動画ファイルのパス
    OUTPUT_FOLDER = "./faces/video"  # 出力フォルダ
    
    # パターン1: フレーム間隔で抽出（例：30フレームごと）
    print("=== フレーム間隔での抽出 ===")
    extract_frames(
        video_path=VIDEO_PATH,
        output_folder=OUTPUT_FOLDER,
        frame_interval=30,  # 30フレームごと
        image_format='jpg'
    )
    
    