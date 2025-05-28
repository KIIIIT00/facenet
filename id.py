from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import faiss
import torch
import glob
from tqdm import tqdm
import random
import csv
import json
from datetime import datetime
import os

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# 結果保存用のディレクトリを作成
output_dir = "face_recognition_results"
image_output_dir = os.path.join(output_dir, "visualization")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(image_output_dir):
    os.makedirs(image_output_dir)

def get_person_train_files(person_name, train_files, train_labels, id2name, name2id):
    """
    特定の人物の学習用ファイル一覧を取得
    """
    person_id = name2id.get(person_name, -1)
    if person_id == -1:
        return []
    
    person_files = []
    for file, label in zip(train_files, train_labels):
        if label == person_id:
            person_files.append(file)
    
    return sorted(person_files)

def calculate_distances_to_person_dataset(test_embedding, person_train_files, mtcnn, resnet, device):
    """
    テスト画像と特定人物の学習データセット各画像との距離を計算
    """
    distances = []
    valid_files = []
    
    for train_file in person_train_files:
        try:
            # 学習画像の特徴量を抽出
            train_img = Image.open(train_file)
            train_img_tensor = mtcnn(train_img)
            
            if train_img_tensor is not None:
                train_img_tensor = train_img_tensor.to(device)
                train_embedding = resnet(train_img_tensor.unsqueeze(0))
                train_embedding_np = train_embedding.cpu().detach().numpy()
                
                # コサイン類似度を計算（内積として）
                similarity = np.dot(test_embedding.flatten(), train_embedding_np.flatten())
                distances.append(float(similarity))
                valid_files.append(train_file)
            else:
                print(f"Warning: Could not extract face from {train_file}")
                
        except Exception as e:
            print(f"Error processing {train_file}: {e}")
    
    return distances, valid_files

def create_person_dataset_visualization(test_results, person_name, person_id, id2name, name2id, 
                                      train_files, train_labels, mtcnn, resnet, device, output_dir):
    """
    特定人物のテスト結果と、その人物の学習データセット全体との距離を可視化
    """
    person_tests = [r for r in test_results if r['test_name'] == person_name]
    
    if not person_tests:
        return
    
    # その人物の学習用ファイルを取得
    person_train_files = get_person_train_files(person_name, train_files, train_labels, id2name, name2id)
    
    if not person_train_files:
        print(f"No training files found for {person_name}")
        return
    
    # テストケースが1つしかない場合は、学習データセット内の相互類似度分析を行う
    if len(person_tests) == 1:
        create_intra_dataset_similarity_analysis(person_name, person_id, person_train_files, 
                                                mtcnn, resnet, device, output_dir)
        test_idx = 0
        test_result = person_tests[0]
    else:
        # 複数のテストケースがある場合は元の処理
        for test_idx, test_result in enumerate(person_tests):
        print(f"Creating visualization for {person_name} - Test {test_idx + 1}")
        
        # テスト画像の特徴量を抽出
        try:
            test_img = Image.open(test_result['test_file'])
            test_img_tensor = mtcnn(test_img)
            
            if test_img_tensor is None:
                print(f"Could not extract face from test image: {test_result['test_file']}")
                continue
                
            test_img_tensor = test_img_tensor.to(device)
            test_embedding = resnet(test_img_tensor.unsqueeze(0))
            test_embedding_np = test_embedding.cpu().detach().numpy()
            
        except Exception as e:
            print(f"Error processing test image {test_result['test_file']}: {e}")
            continue
        
        # その人物の学習データセット各画像との距離を計算
        distances, valid_train_files = calculate_distances_to_person_dataset(
            test_embedding_np, person_train_files, mtcnn, resnet, device
        )
        
        if not distances:
            print(f"No valid training images found for {person_name}")
            continue
        
        # 距離でソート（降順：高い類似度順）
        sorted_indices = np.argsort(distances)[::-1]
        
        # 可視化を作成
        n_train_images = len(valid_train_files)
        n_cols = min(6, n_train_images)  # 最大6列
        n_rows = max(2, (n_train_images + n_cols - 1) // n_cols + 1)  # テスト画像用に1行追加
        
        fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
        fig.suptitle(f'{person_name} - Test Case {test_idx + 1}\nDataset Distance Analysis', 
                    fontsize=16, fontweight='bold')
        
        # テスト画像を上段中央に表示
        ax_test = plt.subplot(n_rows, n_cols, n_cols // 2 + 1)
        ax_test.imshow(test_img)
        ax_test.set_title(f'TEST IMAGE\n{person_name}\nFile: {os.path.basename(test_result["test_file"])}', 
                         fontsize=12, fontweight='bold', color='blue')
        ax_test.axis('off')
        
        # 緑の太い枠で囲む
        for spine in ax_test.spines.values():
            spine.set_edgecolor('blue')
            spine.set_linewidth(4)
        
        # 統計情報を表示
        if len(distances) > 0:
            max_dist = max(distances)
            min_dist = min(distances)
            avg_dist = np.mean(distances)
            
            stats_text = f'Dataset Statistics:\n' \
                        f'Max Distance: {max_dist:.4f}\n' \
                        f'Min Distance: {min_dist:.4f}\n' \
                        f'Avg Distance: {avg_dist:.4f}\n' \
                        f'Total Images: {len(distances)}'
            
            ax_stats = plt.subplot(n_rows, n_cols, n_cols)
            ax_stats.text(0.1, 0.5, stats_text, transform=ax_stats.transAxes, 
                         fontsize=10, verticalalignment='center',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
            ax_stats.set_title('Statistics', fontweight='bold')
            ax_stats.axis('off')
        
        # 学習データセットの各画像を距離順に表示
        for display_idx, sorted_idx in enumerate(sorted_indices):
            if display_idx >= n_cols * (n_rows - 1):  # 表示可能な数を超えた場合
                break
                
            # 2行目以降に配置
            row = (display_idx // n_cols) + 1
            col = display_idx % n_cols
            ax_idx = row * n_cols + col + 1
            
            ax = plt.subplot(n_rows, n_cols, ax_idx)
            
            train_file = valid_train_files[sorted_idx]
            distance = distances[sorted_idx]
            
            try:
                train_img = Image.open(train_file)
                ax.imshow(train_img)
                
                # 距離に応じた色分け
                if distance > avg_dist:
                    border_color = 'green'
                    title_color = 'green'
                else:
                    border_color = 'orange'
                    title_color = 'orange'
                
                # 最高・最低距離の場合は特別な色
                if distance == max_dist:
                    border_color = 'darkgreen'
                    title_color = 'darkgreen'
                elif distance == min_dist:
                    border_color = 'red'
                    title_color = 'red'
                
                # 枠を設定
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(3)
                
                # タイトル設定
                rank_in_dataset = display_idx + 1
                filename = os.path.basename(train_file)
                title_text = f'Rank {rank_in_dataset}\nDist: {distance:.4f}\n{filename}'
                
                ax.set_title(title_text, fontsize=9, color=title_color, fontweight='bold')
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Image Load Error\n{str(e)[:50]}...', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=8)
                ax.set_title(f'Error\nDist: {distance:.4f}', fontsize=9, color='red')
            
            ax.axis('off')
        
        # FAISSでの予測結果も表示（比較用）
        faiss_text = "FAISS Top-10 Results:\n"
        for i, pred in enumerate(test_result['predictions'][:5]):  # Top-5のみ表示
            status = "✓" if pred['is_correct'] else "✗"
            faiss_text += f"{i+1}. {pred['predicted_name']} ({pred['distance']:.4f}) {status}\n"
        
        # 最後の位置に表示
        if n_cols * n_rows > len(sorted_indices) + n_cols:
            ax_faiss = plt.subplot(n_rows, n_cols, n_cols * n_rows)
            ax_faiss.text(0.1, 0.9, faiss_text, transform=ax_faiss.transAxes, 
                         fontsize=9, verticalalignment='top',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
            ax_faiss.set_title('FAISS Comparison', fontweight='bold')
            ax_faiss.axis('off')
        
        plt.tight_layout()
        
        # ファイル名を生成して保存
        safe_person_name = "".join(c for c in person_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{safe_person_name}_dataset_analysis_test_{test_idx+1}.png"
        filepath = os.path.join(output_dir, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
def create_intra_dataset_similarity_analysis(person_name, person_id, person_train_files, 
                                            mtcnn, resnet, device, output_dir):
    """
    学習データセット内での相互類似度分析を作成
    """
    print(f"Creating intra-dataset similarity analysis for {person_name}")
    
    if len(person_train_files) < 2:
        print(f"Not enough training files for {person_name} to perform similarity analysis")
        return
    
    # 全ての学習画像の特徴量を抽出
    embeddings = []
    valid_files = []
    
    for train_file in person_train_files:
        try:
            img = Image.open(train_file)
            img_tensor = mtcnn(img)
            
            if img_tensor is not None:
                img_tensor = img_tensor.to(device)
                embedding = resnet(img_tensor.unsqueeze(0))
                embedding_np = embedding.cpu().detach().numpy().flatten()
                embeddings.append(embedding_np)
                valid_files.append(train_file)
            else:
                print(f"Warning: Could not extract face from {train_file}")
                
        except Exception as e:
            print(f"Error processing {train_file}: {e}")
    
    if len(valid_files) < 2:
        print(f"Not enough valid training images for {person_name}")
        return
    
    # 相互類似度行列を計算
    n_images = len(embeddings)
    similarity_matrix = np.zeros((n_images, n_images))
    
    for i in range(n_images):
        for j in range(n_images):
            if i != j:
                similarity = np.dot(embeddings[i], embeddings[j])
                similarity_matrix[i, j] = similarity
            else:
                similarity_matrix[i, j] = 1.0  # 自分自身は最大類似度
    
    # 各画像について、他の画像との平均類似度を計算
    avg_similarities = []
    for i in range(n_images):
        other_similarities = [similarity_matrix[i, j] for j in range(n_images) if i != j]
        avg_similarities.append(np.mean(other_similarities))
    
    # 平均類似度でソート（高い順）
    sorted_indices = np.argsort(avg_similarities)[::-1]
    
    # 可視化を作成
    n_cols = min(5, n_images)
    n_rows = max(3, (n_images + n_cols - 1) // n_cols + 1)
    
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    fig.suptitle(f'{person_name} - Training Dataset Similarity Analysis\n'
                f'Inter-image similarity within the same person', 
                fontsize=16, fontweight='bold')
    
    # 統計情報を表示
    overall_avg = np.mean([s for s in avg_similarities])
    max_avg = max(avg_similarities)
    min_avg = min(avg_similarities)
    
    ax_stats = plt.subplot(n_rows, n_cols, (n_cols // 2) + 1)
    stats_text = f'Dataset Statistics:\n' \
                f'Images: {n_images}\n' \
                f'Avg Similarity: {overall_avg:.4f}\n' \
                f'Max Avg: {max_avg:.4f}\n' \
                f'Min Avg: {min_avg:.4f}\n' \
                f'Std Dev: {np.std(avg_similarities):.4f}'
    
    ax_stats.text(0.1, 0.5, stats_text, transform=ax_stats.transAxes, 
                 fontsize=12, verticalalignment='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
    ax_stats.set_title('Overall Statistics', fontweight='bold', fontsize=14)
    ax_stats.axis('off')
    
    # 各画像を平均類似度順に表示
    display_count = 0
    for display_idx, sorted_idx in enumerate(sorted_indices):
        if display_count >= n_cols * (n_rows - 1):  # 統計用の行を除く
            break
        
        # 2行目以降に配置
        row = (display_count // n_cols) + 1
        col = display_count % n_cols
        ax_idx = row * n_cols + col + 1
        
        if ax_idx > n_rows * n_cols:
            break
        
        ax = plt.subplot(n_rows, n_cols, ax_idx)
        
        train_file = valid_files[sorted_idx]
        avg_sim = avg_similarities[sorted_idx]
        
        try:
            train_img = Image.open(train_file)
            ax.imshow(train_img)
            
            # 平均類似度に応じた色分け
            if avg_sim > overall_avg + np.std(avg_similarities):
                border_color = 'darkgreen'
                title_color = 'darkgreen'
                quality_label = "High Similarity"
            elif avg_sim > overall_avg:
                border_color = 'green'
                title_color = 'green'
                quality_label = "Above Average"
            elif avg_sim > overall_avg - np.std(avg_similarities):
                border_color = 'orange'
                title_color = 'orange'
                quality_label = "Below Average"
            else:
                border_color = 'red'
                title_color = 'red'
                quality_label = "Low Similarity"
            
            # 枠を設定
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
            
            # その画像と最も類似する他の画像を見つける
            image_similarities = similarity_matrix[sorted_idx]
            best_match_idx = np.argmax([s if i != sorted_idx else -1 for i, s in enumerate(image_similarities)])
            best_match_sim = image_similarities[best_match_idx]
            
            # タイトル設定
            filename = os.path.basename(train_file)
            rank_in_dataset = display_idx + 1
            title_text = f'Rank {rank_in_dataset} - {quality_label}\n' \
                        f'Avg Sim: {avg_sim:.4f}\n' \
                        f'Best Match: {best_match_sim:.4f}\n' \
                        f'{filename}'
            
            ax.set_title(title_text, fontsize=9, color=title_color, fontweight='bold')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Image Load Error\n{str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.set_title(f'Error\nAvg Sim: {avg_sim:.4f}', fontsize=9, color='red')
        
        ax.axis('off')
        display_count += 1
    
    # 類似度ヒートマップを追加（小さく表示）
    if n_images <= 10:  # 画像数が少ない場合のみヒートマップを表示
        ax_heatmap = plt.subplot(n_rows, n_cols, n_cols * n_rows)
        
        # ファイル名を短縮
        short_names = [os.path.basename(f)[:10] + '...' if len(os.path.basename(f)) > 10 
                      else os.path.basename(f) for f in valid_files]
        
        im = ax_heatmap.imshow(similarity_matrix, cmap='coolwarm', vmin=0, vmax=1)
        ax_heatmap.set_xticks(range(n_images))
        ax_heatmap.set_yticks(range(n_images))
        ax_heatmap.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
        ax_heatmap.set_yticklabels(short_names, fontsize=8)
        ax_heatmap.set_title('Similarity Heatmap', fontsize=10, fontweight='bold')
        
        # 数値を表示（小さいフォント）
        for i in range(n_images):
            for j in range(n_images):
                if i != j:  # 対角線以外
                    text = ax_heatmap.text(j, i, f'{similarity_matrix[i, j]:.2f}', 
                                         ha="center", va="center", fontsize=6,
                                         color="white" if similarity_matrix[i, j] < 0.5 else "black")
        
        plt.colorbar(im, ax=ax_heatmap, shrink=0.8)
    
    plt.tight_layout()
    
    # ファイル名を生成して保存
    safe_person_name = "".join(c for c in person_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    filename = f"{safe_person_name}_intra_dataset_similarity.png"
    filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"Saved intra-dataset similarity analysis: {filepath}")
    
    return True

def create_distance_distribution_plot(test_results, id2name, name2id, train_files, train_labels, 
                                     mtcnn, resnet, device, output_dir):
    """
    各人物の学習データセット内距離分布を可視化
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    person_names = list(name2id.keys())[:6]  # 最初の6人のみ
    
    for idx, person_name in enumerate(person_names):
        if idx >= 6:
            break
            
        ax = axes[idx]
        person_tests = [r for r in test_results if r['test_name'] == person_name]
        
        if not person_tests:
            continue
        
        # その人物の学習ファイルを取得
        person_train_files = get_person_train_files(person_name, train_files, train_labels, id2name, name2id)
        
        all_distances = []
        
        # 各テストケースについて学習データセットとの距離を計算
        for test_result in person_tests:
            try:
                test_img = Image.open(test_result['test_file'])
                test_img_tensor = mtcnn(test_img)
                
                if test_img_tensor is not None:
                    test_img_tensor = test_img_tensor.to(device)
                    test_embedding = resnet(test_img_tensor.unsqueeze(0))
                    test_embedding_np = test_embedding.cpu().detach().numpy()
                    
                    distances, _ = calculate_distances_to_person_dataset(
                        test_embedding_np, person_train_files, mtcnn, resnet, device
                    )
                    all_distances.extend(distances)
                    
            except Exception as e:
                print(f"Error in distribution calculation for {person_name}: {e}")
        
        if all_distances:
            ax.hist(all_distances, bins=15, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(all_distances), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(all_distances):.3f}')
            ax.set_title(f'{person_name}\nDataset Distance Distribution')
            ax.set_xlabel('Distance')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{person_name}\n(No Data)')
    
    # 余った軸を非表示
    for idx in range(len(person_names), 6):
        axes[idx].axis('off')
    
    plt.suptitle('Distance Distribution Analysis per Person', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    dist_plot_path = os.path.join(output_dir, "distance_distributions.png")
    plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved distance distribution plot: {dist_plot_path}")

# メインコード部分
files = glob.glob("./input/Original Images/*/*")

labels = []
for file in files:
    s = file.split("/")[-1].split("_")[0]
    labels.append(s)

id2name = dict()
for i,e in enumerate(set(labels)):
    id2name[i] = e
name2id = {id2name[i]:i for i in id2name}

train_files, train_labels = [],[]
for i in range(0, len(files), 2):
    train_files.append(files[i])
    train_labels.append(name2id[labels[i]])
    
test_files, test_labels = [], []
for i in range(1, len(files), 2):
    test_files.append(files[i])
    test_labels.append(name2id[labels[i]])

device = "cpu"

dim = 512
nlist = 10
m = 32
nbits = 5

quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)

mtcnn = MTCNN(image_size=160, margin=10, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').to(device).eval()

print("Extracting embeddings from training data...")
embeddings = None
for file in tqdm(train_files):
    img = Image.open(file)
    img = mtcnn(img).to(device)
    embedding = resnet(img.unsqueeze(0)).cpu().detach().numpy()
    if embeddings is None:
        embeddings = embedding
    else:
        embeddings = np.concatenate((embeddings, embedding), axis=0)

print("Building FAISS index...")
index.train(embeddings)
index.add_with_ids(embeddings, np.array(train_labels))

print("Running face recognition tests...")
all_results = []

for i, (label, file) in enumerate(zip(test_labels, test_files)):
    if i % 50 == 0:
        img = Image.open(file)
        img = mtcnn(img).to(device)
        embedding = resnet(img.unsqueeze(0)).cpu().detach().numpy()
        D, I = index.search(embedding, 10)
        
        print(f"label= {label} {id2name[label]}")
        
        test_result = {
            'test_index': i,
            'test_label': label,
            'test_name': id2name[label],
            'test_file': file,
            'predictions': []
        }
        
        correct_count = 0
        for rank, (d, pred_id) in enumerate(zip(D[0], I[0])):
            is_correct = (label == pred_id)
            if is_correct:
                correct_count += 1
                status = "---OK"
            else:
                status = "xxxNG"
            
            print(f"  distance = {d:.6f} index = {pred_id} {id2name[pred_id]} {status}")
            
            test_result['predictions'].append({
                'rank': rank + 1,
                'distance': float(d),
                'predicted_id': int(pred_id),
                'predicted_name': id2name[pred_id],
                'is_correct': is_correct
            })
        
        test_result['top1_accuracy'] = (label == I[0][0])
        test_result['total_correct_in_top10'] = correct_count
        
        all_results.append(test_result)

print("\nCreating dataset-based visualizations...")

# 人物ごとのデータセット分析可視化を作成
for person_id, person_name in id2name.items():
    create_person_dataset_visualization(
        all_results, person_name, person_id, id2name, name2id,
        train_files, train_labels, mtcnn, resnet, device, image_output_dir
    )

# 距離分布プロットを作成
create_distance_distribution_plot(
    all_results, id2name, name2id, train_files, train_labels,
    mtcnn, resnet, device, image_output_dir
)

print(f"\nAll dataset visualizations saved to: {image_output_dir}")
print("Available files:")
for filename in sorted(os.listdir(image_output_dir)):
    if filename.endswith('.png'):
        print(f"  - {filename}")