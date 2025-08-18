# 01_prepare_dataset_splits.py
import argparse, gzip, ast, json, random, math
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm

ROOT = 'Clothing, Shoes & Jewelry'

def parse_meta_gz(path: str):
    with gzip.open(path, 'rt', encoding='utf-8') as g:
        for line in g:
            line = line.strip()
            if not line:
                continue
            yield ast.literal_eval(line)

def categories_to_tokens(cats_ll: Any) -> List[str]:
    toks = []
    if isinstance(cats_ll, list):
        for sub in cats_ll:
            if not isinstance(sub, list):
                continue
            for c in sub:
                if c and c != ROOT:
                    toks.append(str(c).strip())
    # dedup, preserve order
    seen = set()
    out = []
    for t in toks:
        if t and t not in seen:
            out.append(t); seen.add(t)
    return out

def safe_price(x):
    try:
        p = float(x)
        if math.isfinite(p) and p > 0:
            return p
    except Exception:
        pass
    return float('nan')

def pick_holdout_categories(df: pd.DataFrame, val_ratio: float, seed: int = 42):
    """Chọn 1 tập main_cat để đưa hết vào validation sao cho tổng số dòng ~ val_ratio."""
    rng = random.Random(seed)
    cats = df['main_cat'].dropna().unique().tolist()
    rng.shuffle(cats)
    target = int(len(df) * val_ratio)
    chosen, total = [], 0
    for c in cats:
        n = int((df['main_cat'] == c).sum())
        chosen.append(c)
        total += n
        if total >= target:
            break
    return set(chosen)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta_gz', required=True, help='meta_*.json.gz (UCSD format)')
    ap.add_argument('--img_dir', required=True, help='Folder of resized images (from 00)')
    ap.add_argument('--out_dir', required=True, help='Output folder')
    ap.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no_leakage', action='store_true',
                    help='Nếu bật: chọn nguyên cụm main_cat làm val (không trộn main_cat giữa train/val)')
    ap.add_argument('--min_tokens', type=int, default=2,
                    help='Bỏ item có ít hơn min_tokens (sau khi bỏ ROOT)')
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    rows = []
    img_dir = Path(args.img_dir)

    # Build dataframe
    for obj in tqdm(parse_meta_gz(args.meta_gz), desc='Scanning meta'):
        asin = str(obj.get('asin', '')).strip()
        if not asin:
            continue
        img_path = img_dir / f"{asin}.jpg"
        if not img_path.exists():
            continue

        cats_ll = obj.get('categories', [])
        tokens = categories_to_tokens(cats_ll)

        # tính main_cat = token cuối của path sâu nhất (bỏ ROOT)
        main_cat = ''
        best = []
        if isinstance(cats_ll, list):
            for sub in cats_ll:
                if isinstance(sub, list):
                    filt = [c for c in sub if c != ROOT]
                    if len(filt) > len(best):
                        best = filt
        if best:
            main_cat = str(best[-1]).strip()

        # giữ item có tối thiểu tokens
        if len(tokens) < args.min_tokens:
            continue

        price = safe_price(obj.get('price', float('nan')))  # có thể trống trong meta
        rows.append({
            'asin': asin,
            'image_path': str(img_path),
            'cats_tokens_json': json.dumps(tokens, ensure_ascii=False),
            'main_cat': main_cat,
            'price': price
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit('No rows found. Check --img_dir and meta alignment.')

    # Save all
    df.to_csv(out / 'dataset_all.csv', index=False)

    # Split
    if args.no_leakage:
        holdout = pick_holdout_categories(df, val_ratio=args.val_ratio, seed=args.seed)
        val = df[df['main_cat'].isin(holdout)].copy()
        train = df[~df['main_cat'].isin(holdout)].copy()
    else:
        # Stratified sample by main_cat (có leakage, nhưng cân bằng)
        val = df.groupby('main_cat', group_keys=False).apply(
            lambda g: g.sample(frac=args.val_ratio, random_state=args.seed)
        )
        train = pd.concat([df, val]).drop_duplicates(keep=False)

    train.to_csv(out / 'dataset_train.csv', index=False)
    val.to_csv(out / 'dataset_val.csv', index=False)

    print('Saved:')
    print(' -', out / 'dataset_all.csv')
    print(' -', out / 'dataset_train.csv', f'({len(train)})')
    print(' -', out / 'dataset_val.csv', f'({len(val)})')
    if args.no_leakage:
        print('[INFO] no_leakage: main_cat không trùng giữa train/val')

if __name__ == '__main__':
    main()
