#!/usr/bin/env python
# coding: utf-8
"""
Deep-learning product-retrieval pipeline using pre-trained model with per-run recommendation output
Author : you
Date   : 2025-05-20
"""

import os, json, random, math, argparse, time
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from sentence_transformers import SentenceTransformer

import faiss                  

NUM_TEST          = 50000    
TOP_K             = 100       
NUM_RUNS          = 5       
SEED              = 42
MAX_PER_CATEGORY  = 50_000    
BATCH_ENCODE      = 256     
MODEL_NAME        = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR        = "dl_output5"
USE_GPU           = torch.cuda.is_available() 
DEVICE            = "cuda" if USE_GPU else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def read_jsonl(path: str) -> pd.DataFrame:
    """Đọc file metadata dạng jsonl."""
    return pd.read_json(path, lines=True)

def process_categories_for_df(categories):
    """Xử lý danh mục để lấy danh mục cuối cùng."""
    if categories and isinstance(categories, list):
        filtered = [cat[-1] for cat in categories if cat and cat[0] != "Clothing, Shoes & Jewelry"]
        return filtered[-1] if filtered else categories[-1][-1]
    return ""

def process_salesRank(salesRank):
    """Xử lý salesRank để lấy key đầu tiên."""
    if salesRank and isinstance(salesRank, dict):
        return list(salesRank.keys())[0]
    return ""

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Tiền xử lý dữ liệu DataFrame."""
    df['title']      = df['title'].fillna('').astype(str)
    df['brand']      = df['brand'].fillna('').astype(str)
    df['categories'] = df['categories'].apply(process_categories_for_df)
    df['salesRank']  = df['salesRank'].apply(process_salesRank)
    df['combined']   = df['title'] + " " + df['categories'] + " " + df['salesRank'] + " " + df['brand']
    return df

def load_metadata(path: str) -> dict:
    """Tải metadata dạng asin -> categories."""
    asin_to_cat = {}
    with open(path, 'r') as f:
        for line in f:
            try:
                obj = json.loads(line)
                asin_to_cat[obj["asin"]] = obj.get("categories", [])
            except Exception:
                continue
    return asin_to_cat

def categories_match(cat1, cat2) -> bool:
    """Kiểm tra xem hai danh mục có giao nhau không."""
    def to_set(lst):
        s = set()
        for sub in lst:
            for c in sub:
                if c != "Clothing, Shoes & Jewelry":
                    s.add(c)
        return s
    return len(to_set(cat1) & to_set(cat2)) > 0


def balance_data(df, vectors, asin_to_cat):
    """Cân bằng dữ liệu theo danh mục."""
    asin2main = {}
    for asin, cats in asin_to_cat.items():
        main = None
        max_len = 0
        for sub in cats:
            filt = [c for c in sub if c != "Clothing, Shoes & Jewelry"]
            if filt and len(filt) > max_len:
                main, max_len = filt[-1], len(filt)
        if main:
            asin2main[asin] = main
    counter = Counter(asin2main.values())
    print("Trước khi cân bằng:", counter.most_common(10))

    cat_idxs = {c: [] for c in counter}
    for idx, row in df.iterrows():
        a = row['asin']
        c = asin2main.get(a)
        if c and vectors[idx] is not None:
            cat_idxs[c].append(idx)

    keep = []
    for c, idxs in cat_idxs.items():
        keep.extend(random.sample(idxs, min(len(idxs), MAX_PER_CATEGORY)))
    keep.sort()

    new_df      = df.iloc[keep].reset_index(drop=True)
    new_vectors = [vectors[i] for i in keep]
    print(f"✅ Sau khi cân bằng: {len(new_df)} items")
    return new_df, new_vectors


def encode_corpus(model: SentenceTransformer, texts: list, batch=BATCH_ENCODE):
    """Mã hóa corpus thành embeddings."""
    return model.encode(texts,
                        batch_size=batch,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        device=DEVICE)

def build_faiss(vectors: np.ndarray):
    """Xây dựng chỉ mục FAISS."""
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors.astype(np.float32))
    return index

def search_faiss(index, query_vec, k=TOP_K):
    """Tìm kiếm trên FAISS."""
    D, I = index.search(np.array([query_vec], dtype=np.float32), k+1)
    return I[0][1:], D[0][1:]


def dcg(rels, k):   return sum([(2**r-1)/math.log2(i+2) for i,r in enumerate(rels[:k])])
def idcg(k):        return dcg([1]*k, k)

def evaluate(df, vectors, test_idx, asin_to_cat, index):
    """Đánh giá hiệu suất."""
    P_at = {5:0, 10:0, 20:0}
    ndcg = {5:0, 10:0, 20:0}
    mrr  = 0
    correct = 0; total = 0

    for count, idx in enumerate(test_idx, 1):
        if count % 1000 == 0: print(f"  test {count}/{len(test_idx)}")
        q_vec = vectors[idx]; q_asin = df.iloc[idx]['asin']
        if q_vec is None: continue
        sim_idx, sim_scores = search_faiss(index, q_vec)
        rels = []; first = None
        for rank, (ridx, score) in enumerate(zip(sim_idx, sim_scores)):
            r_asin = df.iloc[ridx]['asin']
            match = categories_match(asin_to_cat.get(q_asin, []), asin_to_cat.get(r_asin, []))
            rels.append(1 if match else 0)
            if match and first is None: first = rank + 1
        for k in P_at:
            P_at[k] += sum(rels[:k])/k if len(rels) >= k else 0
            ndcg[k] += dcg(rels, k)/idcg(k)
        if first: mrr += 1/first
        correct += sum(rels)
        total   += len(rels)

    N = len(test_idx)
    metrics = {
        "Precision@5":  P_at[5]/N,
        "Precision@10": P_at[10]/N,
        "Precision@20": P_at[20]/N,
        "MRR":          mrr/N,
        "NDCG@5":       ndcg[5]/N,
        "NDCG@10":      ndcg[10]/N,
        "NDCG@20":      ndcg[20]/N,
        "Accuracy":     correct/total if total else 0
    }
    return metrics

def plot_bar(metrics: dict, title: str, path: str):
    """Vẽ biểu đồ cột."""
    plt.figure(figsize=(10, 5))
    labels, vals = list(metrics.keys()), list(metrics.values())
    bars = plt.bar(labels, vals)
    plt.ylim(0, 1); plt.xticks(rotation=45); plt.title(title)
    for b in bars:
        plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f"{b.get_height():.3f}",
                 ha='center', va='bottom', fontsize=8)
    plt.tight_layout(); plt.savefig(path); plt.close()

# ============== 7. RECOMMENDATION GENERATION ======

# def generate_recommendations(df, vectors, test_idx, index, run):
#     """Tạo file gợi ý cho tập test của mỗi run."""
#     recommendations = []
#     for idx in test_idx:
#         asin = df.iloc[idx]['asin']
#         q_vec = vectors[idx]
#         if q_vec is None: continue
#         sim_idx, sim_scores = search_faiss(index, q_vec)
#         rec_asins = [df.iloc[ridx]['asin'] for ridx in sim_idx[:TOP_K]]
#         avg_similarity = np.mean(sim_scores[:TOP_K])
#         recommendations.append({
#             'asin': asin,
#             'recommend_asins': ','.join(rec_asins),
#             'avg_similarity': avg_similarity
#         })
#     rec_df = pd.DataFrame(recommendations)
#     output_path = os.path.join(OUTPUT_DIR, f'recommendations_run_{run}.csv')
#     rec_df.to_csv(output_path, index=False)
#     print(f"✅ Đã lưu file gợi ý cho run {run} vào {output_path}")


def main(jsonl_path):
    t0 = time.time()
    print("🔹 Đọc file...")
    df = preprocess_df(read_jsonl(jsonl_path))
    asin_to_cat = load_metadata(jsonl_path)

    print("🔹 Tải mô hình pre-trained (không fine-tune)...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    print("🔹 Mã hóa corpus...")
    vectors = encode_corpus(model, df['combined'].tolist())
    df, vectors = balance_data(df, list(vectors), asin_to_cat)

    print("🔹 Xây dựng chỉ mục FAISS...")
    index = build_faiss(np.vstack([v for v in vectors if v is not None]))

    if NUM_TEST > len(df):
        raise ValueError(f"NUM_TEST ({NUM_TEST}) cannot be larger than the number of products ({len(df)}).")

    all_metrics = []
    for run in range(1, NUM_RUNS + 1):
        print(f"\n=== RUN {run}/{NUM_RUNS} ===")
        rng = np.random.default_rng(SEED + run)
        test_idx = rng.choice(len(df), size=NUM_TEST, replace=False)
        m = evaluate(df, vectors, test_idx, asin_to_cat, index)
        all_metrics.append(m)
        for k, v in m.items(): print(f"{k}: {v:.4f}")
        plot_bar(m, f"Run {run}", f"{OUTPUT_DIR}/metrics_run_{run}.png")
        print(f"🔹 Tạo file gợi ý cho run {run}...")
        # generate_recommendations(df, vectors, test_idx, index, run)

    avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    print("\n=== TRUNG BÌNH ===")
    for k, v in avg.items(): print(f"{k}: {v:.4f}")
    plot_bar(avg, "Average metrics", f"{OUTPUT_DIR}/metrics_avg.png")
    pd.DataFrame([avg]).to_csv(f"{OUTPUT_DIR}/avg_metrics.csv", index=False)

    print(f"\n✅ Hoàn tất - thời gian {(time.time() - t0)/60:.1f} phút")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default=r"C:\Users\lethi\PycharmProjects\PythonProject4\CutFile\meta_Clothing_Shoes_and_Jewelry_clean\meta_Clothing_Shoes_and_Jewelry_clean.jsonl")
    args = parser.parse_args()
    main(args.jsonl)