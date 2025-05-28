from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import cv2

mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# 画像ファイルから画像の特徴ベクトルを取得(ndarray 512次元)
def feature_vector(image_path):
    img = Image.open(image_path)
    img_cropped = mtcnn(img)
    feature_vector = resnet(img_cropped.unsqueeze(0))
    feature_vector_np = feature_vector.squeeze().to('cpu').detach().numpy().copy()
    return feature_vector_np

# 2つのベクトル間のコサイン類似度を取得
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 画像ファイルを読み込み
img_fv = []
video_name = "video"
face_img_fold = "./faces/" + video_name
print(f"dir exit: {os.path.exists(face_img_fold)}")
os.chdir(face_img_fold)
src_files = glob.glob("*.jpg")

for i, file in enumerate(src_files):
    # image_path = face_img_fold + "/" + file
    image_path = file
    print(os.path.isfile(image_path))
    img = Image.open(image_path)
    output_path = f"{video_name}_{i:02d}.jpg"
    img.save(output_path, 'JPEG')
    img_fv.append(feature_vector(image_path))

# 2枚の画像間の類似度を取得
for i, fv1 in enumerate(img_fv):
  for j, fv2 in enumerate(img_fv):
    similarity = cosine_similarity(fv1, fv2)
    print(f"{i}-{j}：{similarity}")
  print("------------------------")
