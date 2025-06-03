import pandas as pd
import numpy as np
import torch
from gensim.models import FastText
import faiss
import os
import random
from collections import Counter
import matplotlib.pyplot as plt
import json

NUM_TEST = 50000  
TOP_K = 100       
NUM_RUNS = 10      
SEED = 42        
OUTPUT_CSV = "recommendations_fastText_with_metrics_10.csv"
USE_GPU = False   
MAX_PER_CATEGORY = 50000  

def read_jsonl(file_path):
    return pd.read_json(file_path, lines=True)

def process_categories_for_df(categories):
    if categories and isinstance(categories, list):
        filtered = [cat[-1] for cat in categories if cat and cat[0] != "Clothing, Shoes & Jewelry"]
        return filtered[-1] if filtered else categories[-1][-1]
    return ""

def process_salesRank(salesRank):
    if salesRank and isinstance(salesRank, dict):
        return list(salesRank.keys())[0]
    return ""

def preprocess_data(df):
    df['title'] = df['title'].fillna('').astype(str)
    df['brand'] = df['brand'].fillna('').astype(str)
    df['categories'] = df['categories'].apply(process_categories_for_df)
    df['salesRank'] = df['salesRank'].apply(process_salesRank)
    df['combined_text'] = df['title'] + " " + df['categories'] + " " + df['salesRank'] + " " + df['brand']
    return df

def create_fasttext_vectors(df, vector_size=100, window=5, min_count=3):
    sentences = [row['combined_text'].split() for _, row in df.iterrows()]
    model = FastText(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=4)
    vectors = []
    for sentence in sentences:
        word_vectors = [model.wv[word] for word in sentence if word in model.wv]
        if word_vectors:
            vectors.append(np.mean(word_vectors, axis=0, dtype=np.float32))
        else:
            vectors.append(None)
    return vectors, model

def balance_data(df, vectors, asin_to_categories, max_per_category=MAX_PER_CATEGORY):
    """Giới hạn số lượng sản phẩm trong mỗi danh mục để cân bằng dữ liệu."""
    asin_to_main_category = {}
    for asin, categories in asin_to_categories.items():
        main_category = None
        max_length = 0
        for category_list in categories:
            filtered_list = [cat for cat in category_list if cat != "Clothing, Shoes & Jewelry"]
            if filtered_list and len(filtered_list) > max_length:
                main_category = filtered_list[-1]
                max_length = len(filtered_list)
        if main_category:
            asin_to_main_category[asin] = main_category

    category_counts = Counter(asin_to_main_category.values())
    print("Phân bố danh mục trước khi cân bằng:", category_counts)

    filtered_indices = []
    category_indices = {cat: [] for cat in category_counts}
    for idx, row in df.iterrows():
        asin = row['asin']
        cat = asin_to_main_category.get(asin)
        if cat and vectors[idx] is not None:
            category_indices[cat].append(idx)

    for cat, indices in category_indices.items():
        if len(indices) > max_per_category:
            indices = random.sample(indices, max_per_category)
        filtered_indices.extend(indices)

    filtered_indices = sorted(filtered_indices)
    filtered_df = df.iloc[filtered_indices].reset_index(drop=True)
    filtered_vectors = [vectors[i] for i in filtered_indices]
    print(f"✅ Đã cân bằng dữ liệu: {len(filtered_df)} sản phẩm")
    return filtered_df, filtered_vectors, asin_to_main_category

def build_faiss_index(vectors):
    valid_vectors = np.array([v for v in vectors if v is not None], dtype=np.float32)
    dimension = valid_vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)  
    index.add(valid_vectors)
    return index

def find_similar_products_faiss(index, vectors, query_index, top_n=TOP_K):
    query_vector = vectors[query_index]
    if query_vector is None:
        return [], []
    query_vector = np.array([query_vector], dtype=np.float32)
    distances, indices = index.search(query_vector, top_n + 1)  
    similar_indices = indices[0][1:]  
    sim_scores = distances[0][1:]
    return similar_indices, sim_scores

def calculate_dcg(relevances, k):
    return np.sum([(2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(relevances[:k])])

def calculate_idcg(k):
    return calculate_dcg([1] * k, k)

def categories_match(cat_list1, cat_list2):
    set1 = set()
    set2 = set()
    for cat_sublist in cat_list1:
        for cat in cat_sublist:
            if cat != "Clothing, Shoes & Jewelry":
                set1.add(cat)
    for cat_sublist in cat_list2:
        for cat in cat_sublist:
            if cat != "Clothing, Shoes & Jewelry":
                set2.add(cat)
    return len(set1.intersection(set2)) > 0

def evaluate_metrics(filtered_df, vectors, test_indices, asin_to_categories, index):
    detailed_results = []

    precision_at_k = {5: 0, 10: 0, 20: 0}
    mrr = 0
    ndcg_at_k = {5: 0, 10: 0, 20: 0}
    correct_recommendations = 0
    total_recommendations = 0

    num_test = len(test_indices)
    for i, idx in enumerate(test_indices):
        if i % 1000 == 0:
            print(f"Đang xử lý sản phẩm thứ {i}/{num_test}")

        try:
            similar_indices, sim_scores = find_similar_products_faiss(index, vectors, idx)
            if len(similar_indices) == 0:
                continue

            test_asin = filtered_df.iloc[idx]['asin']
            test_categories = asin_to_categories.get(test_asin)
            if not test_categories:
                continue

            recommended_asins = [filtered_df.iloc[rec_idx]['asin'] for rec_idx in similar_indices]

            correct_count = 0
            relevances = []
            first_correct_rank = None
            for j, rec_idx in enumerate(similar_indices):
                rec_asin = filtered_df.iloc[rec_idx]['asin']
                rec_categories = asin_to_categories.get(rec_asin)
                if not rec_categories:
                    continue
                is_correct = categories_match(test_categories, rec_categories)
                relevances.append(1 if is_correct else 0)
                if is_correct:
                    correct_count += 1
                    if first_correct_rank is None:
                        first_correct_rank = j + 1

            precision_k = {}
            for k in [5, 10, 20]:
                precision_k[k] = sum(relevances[:k]) / k if len(relevances) >= k else 0

            mrr_product = 1 / first_correct_rank if first_correct_rank else 0

            ndcg_k = {}
            for k in [5, 10, 20]:
                dcg = calculate_dcg(relevances, k)
                idcg = calculate_idcg(k)
                ndcg_k[k] = dcg / idcg if idcg > 0 else 0

            accuracy_product = sum(relevances) / len(relevances) if relevances else 0

            detailed_results.append({
                'test_asin': test_asin,
                'recommended_asins': ','.join(recommended_asins),
                'Precision@5': precision_k[5],
                'Precision@10': precision_k[10],
                'Precision@20': precision_k[20],
                'MRR': mrr_product,
                'NDCG@5': ndcg_k[5],
                'NDCG@10': ndcg_k[10],
                'NDCG@20': ndcg_k[20],
                'Accuracy': accuracy_product
            })

            for k in precision_at_k:
                precision_at_k[k] += precision_k[k]
            mrr += mrr_product
            for k in ndcg_at_k:
                ndcg_at_k[k] += ndcg_k[k]
            correct_recommendations += sum(relevances)
            total_recommendations += len(relevances)

        except Exception as e:
            print(f"Lỗi khi xử lý sản phẩm tại chỉ số {idx}: {e}")
            continue

    metrics = {
        'Precision@5': precision_at_k[5] / num_test,
        'Precision@10': precision_at_k[10] / num_test,
        'Precision@20': precision_at_k[20] / num_test,
        'MRR': mrr / num_test,
        'NDCG@5': ndcg_at_k[5] / num_test,
        'NDCG@10': ndcg_at_k[10] / num_test,
        'NDCG@20': ndcg_at_k[20] / num_test,
        'Accuracy': correct_recommendations / total_recommendations if total_recommendations > 0 else 0
    }

    return metrics, pd.DataFrame(detailed_results)

def load_metadata(file_path):
    asin_to_categories = {}
    with open(file_path, 'r') as f:
        for line in f:
            try:
                product = json.loads(line.strip())
                asin = product.get('asin')
                categories = product.get('categories', [])
                if asin and categories:
                    asin_to_categories[asin] = categories
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line}")
                continue
    return asin_to_categories

def plot_metrics(metrics, run_id=None):
    """Vẽ biểu đồ cột cho các chỉ số và lưu thành file PNG."""
    labels = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, values, color='skyblue')
    plt.xlabel('Chỉ số')
    plt.ylabel('Giá trị')
    plt.title(f'Chỉ số đánh giá - {"Trung bình" if run_id is None else f"Lần chạy {run_id}"}')
    plt.ylim(0, 1) 
    plt.xticks(rotation=45)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom')

    if run_id is None:
        plt.savefig('metrics_average10_fastText.png', bbox_inches='tight')
    else:
        plt.savefig(f'metrics_run10_{run_id}_fastText.png', bbox_inches='tight')
    plt.close()

def main(file_path):
    print("Đang đọc dữ liệu...")
    df = read_jsonl(file_path)
    print("Đang tiền xử lý dữ liệu...")
    df = preprocess_data(df)
    print("Đang tạo vector đặc trưng bằng FastText...")
    vectors, model = create_fasttext_vectors(df)
    print("Đang tải metadata...")
    asin_to_categories = load_metadata(file_path)

    print("Đang cân bằng dữ liệu...")
    filtered_df, filtered_vectors, asin_to_main_category = balance_data(df, vectors, asin_to_categories)

    print("Đang xây dựng FAISS index...")
    index = build_faiss_index(filtered_vectors)
    print("FAISS index đã được xây dựng.")

    all_metrics = []
    all_detailed_results = []  

    for run_id in range(1, NUM_RUNS + 1):
        print(f"\n--- Bắt đầu lần chạy {run_id} ---")
        np.random.seed(SEED + run_id)
        test_indices = np.random.choice(len(filtered_df), NUM_TEST, replace=False)

        metrics, detailed_results = evaluate_metrics(filtered_df, filtered_vectors, test_indices, asin_to_categories, index)
        all_metrics.append(metrics)
        all_detailed_results.append(detailed_results)
        plot_metrics(metrics, run_id)  
        print(f"Kết quả lần chạy {run_id}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    plot_metrics(avg_metrics)  
    print("\n--- Kết quả trung bình ---")
    for metric_name, value in avg_metrics.items():
        print(f"{metric_name}: {value:.4f}")

    pd.DataFrame([avg_metrics]).to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Đã lưu kết quả trung bình vào {OUTPUT_CSV}")

   
    detailed_df = pd.concat(all_detailed_results, ignore_index=True)
    detailed_df.to_csv('detailed_recommendations.csv', index=False)
    print("✅ Đã lưu chi tiết gợi ý vào detailed_recommendations.csv")

if __name__ == "__main__":
    file_path = "./CutFile/meta_Clothing_Shoes_and_Jewelry_clean/meta_Clothing_Shoes_and_Jewelry_clean.jsonl"
    main(file_path)