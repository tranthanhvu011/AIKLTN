# 07_eval_on_index.py
import argparse, json, math
from pathlib import Path
import numpy as np, pandas as pd
from tqdm import tqdm
import faiss

# plotting (tự động bỏ qua nếu thiếu matplotlib)
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyArrow
    HAS_MPL = True
except Exception:
    HAS_MPL = False

ROOT = 'Clothing, Shoes & Jewelry'

def categories_overlap_ge2(c1, c2):
    """Relevance: 2 sản phẩm 'giống' nếu overlap >= 2 token danh mục (bỏ ROOT)."""
    s1, s2 = set(c1), set(c2)
    if ROOT in s1: s1.remove(ROOT)
    if ROOT in s2: s2.remove(ROOT)
    return len(s1 & s2) >= 2

def load_cats_dict(csv_path):
    """Đọc cats từ CSV -> dict: asin -> list token danh mục."""
    df = pd.read_csv(csv_path)
    if 'cats_tokens_json' not in df.columns or 'asin' not in df.columns:
        raise ValueError("CSV phải có cột 'asin' và 'cats_tokens_json'.")
    cats = df[['asin','cats_tokens_json']].dropna().copy()
    cats['cats'] = cats['cats_tokens_json'].apply(lambda s: json.loads(s))
    return dict(zip(cats['asin'], cats['cats']))

def _safe_savefig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def _draw_pipeline_diagram(out_path, index_size, metric, topk_list):
    if not HAS_MPL: return
    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    ax.axis('off')

    boxes = {
        "Query\nembedding": (0.05, 0.55, 0.22, 0.25),
        f"FAISS Index\n({index_size:,} items)\nmetric={metric}": (0.35, 0.55, 0.30, 0.25),
        f"Top-K\nK={max(topk_list)}": (0.72, 0.55, 0.20, 0.25),
        "Metrics\nRecall@K / Precision@K\nMRR, mAP@10, nDCG@10": (0.35, 0.10, 0.57, 0.28),
    }

    rects = {}
    for label, (x, y, w, h) in boxes.items():
        r = Rectangle((x, y), w, h, fill=False, linewidth=1.5)
        ax.add_patch(r)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=10)
        rects[label] = r

    def mid_right(r): x,y=r.get_xy(); w,h=r.get_width(),r.get_height(); return (x+w, y+h/2)
    def mid_left(r): x,y=r.get_xy(); h=r.get_height(); return (x, y+h/2)

    q = rects["Query\nembedding"]
    idx = rects[f"FAISS Index\n({index_size:,} items)\nmetric={metric}"]
    tk = rects[f"Top-K\nK={max(topk_list)}"]
    met = rects["Metrics\nRecall@K / Precision@K\nMRR, mAP@10, nDCG@10"]

    def arrow(p0, p1):
        ax.add_patch(FancyArrow(p0[0], p0[1], p1[0]-p0[0], p1[1]-p0[1],
                                width=0.002, head_width=0.02, length_includes_head=True))

    arrow(mid_right(q), mid_left(idx))
    arrow(mid_right(idx), mid_left(tk))
    arrow((tk.get_x()+tk.get_width()/2, tk.get_y()), mid_right(met))

    _safe_savefig(fig, out_path)

def _plot_all_metrics_one_figure(out: dict, topk_list, out_path: Path):
    """Gộp tất cả chỉ số vào 1 biểu đồ bar duy nhất."""
    if not HAS_MPL:
        return

    labels, values = [], []
    ks = sorted(set(topk_list))
    for k in ks:
        labels.append(f"R@{k}"); values.append(out.get(f"Recall@{k}", float('nan')))
        labels.append(f"P@{k}"); values.append(out.get(f"Precision@{k}", float('nan')))
        if f"Recall_strict@{k}" in out:  # nếu có bật strict
            labels.append(f"R_strict@{k}"); values.append(out[f"Recall_strict@{k}"])
        if f"F1@{k}" in out:
            labels.append(f"F1@{k}"); values.append(out[f"F1@{k}"])

    # Thêm các global metrics
    for disp, key in [("MRR", "MRR"), ("mAP@10", "mAP@10"), ("nDCG@10", "nDCG@10")]:
        if key in out:
            labels.append(disp); values.append(out[key])

    fig_w = max(10, 0.55 * len(labels))  # rộng hơn nếu nhiều cột
    fig = plt.figure(figsize=(fig_w, 5.5))
    ax = fig.add_subplot(111)
    xs = np.arange(len(labels))
    ax.bar(xs, values)
    ax.set_title("Tổng hợp tất cả chỉ số (gộp 1 biểu đồ)")
    ax.set_ylabel("Giá trị (0–1)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylim(0.0, 1.05)  # thang 0–1

    ax.grid(axis='y', linestyle='--', alpha=0.3)
    for x, v in zip(xs, values):
        if isinstance(v, (int, float)) and np.isfinite(v):
            ax.text(x, v, f"{v:.3f}", ha='center', va='bottom', fontsize=9)

    _safe_savefig(fig, out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--emb', required=True, help='emb_out/embeddings.npy')
    ap.add_argument('--map', required=True, help='emb_out/mapping.csv (phải có cột asin)')
    ap.add_argument('--val_csv', required=True, help='data_splits/dataset_val.csv')
    ap.add_argument('--all_csv', required=True, help='data_splits/dataset_all.csv (để lấy cats cho toàn catalog)')
    ap.add_argument('--index_csv', default='', help='CSV để build index (mặc định = all_csv). '
                                                   'Có thể đặt = data_splits/dataset_train.csv để tránh leakage.')
    ap.add_argument('--topk', nargs='+', type=int, default=[1, 10, 50])
    ap.add_argument('--metric', choices=['cosine', 'l2'], default='cosine')
    ap.add_argument('--with_strict_recall', action='store_true',
                    help='Tính Recall_strict@k = TP@k / (#relevant toàn index) và F1@k.')
    ap.add_argument('--plot_dir', default='', help='Thư mục để lưu hình (PNG). Bỏ trống để không vẽ.')
    ap.add_argument('--save_csv', default='', help='Ghi bảng tổng hợp metric ra CSV (tùy chọn).')
    args = ap.parse_args()

    print('[INFO] Loading embeddings & mapping...')
    E = np.load(args.emb)   # shape (N, D)
    map_df = pd.read_csv(args.map)
    if 'asin' not in map_df.columns:
        raise ValueError("mapping.csv phải có cột 'asin'.")
    asins_all = map_df['asin'].tolist()
    asin2row = {a: i for i, a in enumerate(asins_all)}

    print('[INFO] Loading categories...')
    asin2cats = load_cats_dict(args.all_csv)

    # ----- Build index set (TRAIN-only or ALL) -----
    index_csv = args.index_csv or args.all_csv
    idx_df = pd.read_csv(index_csv)
    if 'asin' not in idx_df.columns:
        raise ValueError("index_csv phải có cột 'asin'.")
    index_asins = [a for a in idx_df['asin'].tolist() if a in asin2row]
    index_rows = np.array([asin2row[a] for a in index_asins], dtype=np.int64)

    E_index = E[index_rows].astype('float32').copy()
    if args.metric == 'cosine':
        faiss.normalize_L2(E_index)
        index = faiss.IndexFlatIP(E_index.shape[1])
    else:
        index = faiss.IndexFlatL2(E_index.shape[1])
    index.add(E_index)
    print(f"[INFO] FAISS index built: {len(index_asins)} items")

    # ----- Prepare queries from VAL -----
    val_df = pd.read_csv(args.val_csv)
    if 'asin' not in val_df.columns:
        raise ValueError("val_csv phải có cột 'asin'.")
    query_asins = [a for a in val_df['asin'].tolist() if a in asin2row]
    print(f"[INFO] Queries (val) usable: {len(query_asins)}")

    # ----- Eval loop -----
    Kmax = max(args.topk) + 5  # dự phòng filter self/thiếu cats
    hits = {k: 0 for k in args.topk}         # HitRate/Recall@k
    prec_sum = {k: 0.0 for k in args.topk}   # Precision@k trung bình

    mrr = 0.0
    map10 = 0.0
    ndcg10 = 0.0

    rec_strict_sum = {k: 0.0 for k in args.topk} if args.with_strict_recall else None
    f1_sum = {k: 0.0 for k in args.topk} if args.with_strict_recall else None

    def dcg(x):  # x: list[int] 0/1
        return sum(((2**xi - 1) / math.log2(i + 2) for i, xi in enumerate(x)))

    n_valid = 0
    for a in tqdm(query_asins, desc='Eval on index'):
        qi = asin2row[a]
        # vector truy vấn
        if args.metric == 'cosine':
            q = E[qi:qi+1].astype('float32').copy()
            faiss.normalize_L2(q)
        else:
            q = E[qi:qi+1].astype('float32', copy=True)

        # Tìm lân cận
        D, I = index.search(q, Kmax)   # (1, Kmax)
        cands = []
        for j in I[0]:
            if j < 0: continue
            cand_asin = index_asins[j]
            if cand_asin == a:  # self-match
                continue
            cands.append(cand_asin)
            if len(cands) >= Kmax - 1:
                break

        # relevance theo overlap danh mục
        qc = asin2cats.get(a)
        if qc is None:
            continue
        rel = []
        for ca in cands[:Kmax - 1]:
            cc = asin2cats.get(ca)
            rel.append(1 if (cc is not None and categories_overlap_ge2(qc, cc)) else 0)

        if not rel:
            continue

        n_valid += 1

        # HitRate-style Recall@k
        for k in args.topk:
            if any(rel[:k]):
                hits[k] += 1

        # Precision@k
        for k in args.topk:
            prec_k = sum(rel[:k]) / max(1, k)
            prec_sum[k] += prec_k

        # (tuỳ chọn) Recall_strict@k & F1@k
        if args.with_strict_recall:
            total_rel = 0
            for cand_asin in index_asins:
                if cand_asin == a:
                    continue
                cc = asin2cats.get(cand_asin)
                if cc is not None and categories_overlap_ge2(qc, cc):
                    total_rel += 1

            for k in args.topk:
                tp_k = sum(rel[:k])
                p_k = tp_k / max(1, k)
                r_k = tp_k / max(1, total_rel)
                rec_strict_sum[k] += r_k
                f1 = (2 * p_k * r_k / (p_k + r_k)) if (p_k + r_k) > 0 else 0.0
                f1_sum[k] += f1

        # MRR
        rr = 0.0
        for rank, r in enumerate(rel, start=1):
            if r == 1:
                rr = 1.0 / rank
                break
        mrr += rr

        # mAP@10
        num_rel, prec_acc = 0, 0.0
        for rank, r in enumerate(rel[:10], start=1):
            if r == 1:
                num_rel += 1
                prec_acc += num_rel / rank
        map10 += (prec_acc / max(1, num_rel)) if num_rel > 0 else 0.0

        # nDCG@10
        ideal = [1] * min(10, sum(rel[:10]))
        ndcg10 += (dcg(rel[:10]) / max(1e-9, dcg(ideal))) if ideal else 0.0

    if n_valid == 0:
        print("[WARN] Không có query hợp lệ để tính metric.")
        return

    # Kết quả
    out = {f"Recall@{k}": hits[k] / n_valid for k in args.topk}
    out['MRR'] = mrr / n_valid
    out['mAP@10'] = map10 / n_valid
    out['nDCG@10'] = ndcg10 / n_valid
    out['queries_evaled'] = n_valid

    for k in args.topk:
        out[f'Precision@{k}'] = prec_sum[k] / n_valid

    print(json.dumps(out, ensure_ascii=False, indent=2))

    # ---- Lưu biểu đồ / file ----
    if args.plot_dir:
        if not HAS_MPL:
            print("[WARN] matplotlib chưa cài, bỏ qua vẽ hình. Cài: pip install matplotlib")
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

        # metrics.json
        with open(plot_dir / 'metrics.json', 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        # CSV tổng hợp (tuỳ chọn)
        if args.save_csv:
            rows = []
            for k in sorted(set(args.topk)):
                rows.append({'metric': f'Recall@{k}', 'value': out.get(f'Recall@{k}', float('nan'))})
                rows.append({'metric': f'Precision@{k}', 'value': out.get(f'Precision@{k}', float('nan'))})
            for name in ['MRR','mAP@10','nDCG@10','queries_evaled']:
                if name in out:
                    rows.append({'metric': name, 'value': out[name]})
            pd.DataFrame(rows).to_csv(args.save_csv, index=False)

        # ✅ 1 biểu đồ duy nhất cho tất cả chỉ số
        _plot_all_metrics_one_figure(out, args.topk, plot_dir / 'metrics_all_in_one.png')

        # Sơ đồ pipeline (tuỳ chọn, vẫn xuất)
        ks = sorted(set(args.topk))
        _draw_pipeline_diagram(plot_dir / 'pipeline_diagram.png', len(index_asins), args.metric, ks)

if __name__ == '__main__':
    main()
