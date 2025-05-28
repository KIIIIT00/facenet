# Face Recognition and Analysis System

このリポジトリは、FaceNet PyTorchとFAISSを使用した顔認識・分析システムです。顔検出、特徴抽出、顔認識、および顔トラッキング機能を提供し、認識結果の詳細な視覚化ツールも含まれています。

## 主な機能

- **顔検出と特徴抽出**: MTCNNとInceptionResnetV1を使用
- **顔認識**: FAISSインデックスを使用した高速顔認識
- **顔トラッキング**: ビデオ内の顔を検出・追跡
- **データセット分析**: 顔認識データセットの類似度分析と視覚化
- **フレーム抽出**: ビデオからの画像フレーム抽出

## 環境構築

### 前提条件

- Python 3.6以上
- pip

### インストール手順

1. リポジトリをクローン：

```bash
$ git clone https://github.com/KIIIIT00/facenet.git
$ cd face-recognition-system
```
2. 仮想環境の作成：
```bash
$ python3 -m venv facenet
$ souce facenet/bin/activate
```
3. 依存関係のインストール：

```bash
$ pip install -r requirements.txt
```

注意: FAISSのインストールに問題がある場合は、[FAISS公式サイト](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)を参照してください。

## データセットの準備

### データセット構造

顔画像データセットは以下の構造で配置します：

```
input/
  Original Images/
    person1_name/
      person1_name_0.jpg
      person1_name_1.jpg
      ...
    person2_name/
      person2_name_0.jpg
      person2_name_1.jpg
      ...
    ...
```

### データセットのダウンロード

公開データセットを使用する場合
```bash
$ curl -L -o ./face-recognition-dataset.zip https://www.kaggle.com/api/v1/datasets/download/vasukipatel/face-recognition-dataset

$ unzip face-recognition-dataset.zip 
```

## 使用方法

### 顔認識の実行

```bash
$ python id.py
```

これにより、データセット内の顔認識処理が実行され、結果が `face_recognition_results/` ディレクトリに保存されます。

### ビデオからの顔トラッキング

1. ビデオファイルを `faces/video.mp4` として配置
2. 以下のコマンドを実行：

```bash
$ python tracking.py
```

処理結果は `video_tracked.mp4` として保存されます。

### ビデオからのフレーム抽出

```bash
$ python videotoimages.py
```

デフォルトでは、`faces/video.mp4` から30フレームごとに画像を抽出し、`faces/video/` ディレクトリに保存します。

### 顔特徴ベクトルの類似度計算

```bash
$ python exmaple.py
```

特定の顔画像間のコサイン類似度を計算します。事前に `faces/video/` ディレクトリに顔画像を配置する必要があります。

## ディレクトリ構造

```
.
├── id.py                      # メイン顔認識スクリプト
├── tracking.py                # 顔トラッキングスクリプト
├── videotoimages.py           # ビデオフレーム抽出スクリプト
├── exmaple.py                 # 顔特徴ベクトル類似度計算スクリプト
├── requirements.txt           # 依存パッケージリスト
├── input/                     # 入力データセット
│   └── Original Images/
│       └── [person_folders]/
├── faces/                     # 顔画像とビデオ
│   ├── video.mp4              # 入力ビデオ
│   └── video/                 # 抽出されたフレーム
└── face_recognition_results/  # 認識結果と視覚化
    └── visualization/
```

## 視覚化の例

顔認識結果の視覚化により、以下のような分析が可能です：

- 個人内データセット類似度分析
- テスト画像と学習データセット間の距離分析
- 顔認識結果の精度評価

視覚化結果は `face_recognition_results/visualization/` ディレクトリに保存されます。

## 注意事項

- このシステムは研究・教育目的で提供されています
- 顔認識技術の使用にはプライバシーへの配慮が必要です
- 大規模なデータセットや高解像度ビデオを処理する場合、十分なメモリとGPUリソースが必要になることがあります

## 依存ライブラリ

- numpy
- Pillow
- requests
- torch
- torchvision
- tqdm
- matplotlib
- faiss-cpu
- facenet-pytorch
