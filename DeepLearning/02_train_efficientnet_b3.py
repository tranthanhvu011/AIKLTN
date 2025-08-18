# 02_train_efficientnet_b3.py
import os, json, math, argparse, random, time, platform, itertools
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

ROOT = 'Clothing, Shoes & Jewelry'

# -------- Utils --------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.benchmark = True

def l2norm(x: torch.Tensor, eps: float = 1e-8):
    return x / (x.norm(dim=1, keepdim=True) + eps)

def get_lr(optimizer):
    for pg in optimizer.param_groups:
        return pg.get('lr', None)

def gpu_mem():
    if torch.cuda.is_available():
        return f"{torch.cuda.memory_allocated() / (1024**2):.0f}MB"
    return "CPU"

def price_close_log(pi: float, pj: float, delta: float) -> bool:
    if not (pi > 0 and pj > 0):
        return False
    return abs(math.log(pi) - math.log(pj)) <= delta

def price_affinity_torch(pi: torch.Tensor, pj: torch.Tensor, sigma: float) -> torch.Tensor:
    # Gaussian in log-space
    # pi, pj shape: (B,)
    pi = pi.clamp_min(1e-6); pj = pj.clamp_min(1e-6)
    r = torch.log(pi) - torch.log(pj)
    return torch.exp(-(r*r) / (2.0 * (sigma**2)))  # in [0, 1]

# -------- Dataset with on-the-fly positive sampling --------
class PairDataset(Dataset):
    def __init__(self, csv_path: str, img_size: int = 300, augment: bool = True,
                 pos_overlap_k: int = 2, pos_price_delta: float = 0.0):
        """
        pos_overlap_k: số token overlap yêu cầu khi chọn positive (>=2 khuyến nghị)
        pos_price_delta: ngưỡng |log(p_i)-log(p_j)| <= delta để coi là 'gần giá' (0 = tắt ràng buộc giá)
        """
        self.df = pd.read_csv(csv_path)
        self.paths = self.df['image_path'].tolist()
        self.asins = self.df['asin'].tolist()
        self.cats = [json.loads(s) for s in self.df['cats_tokens_json'].tolist()]
        self.prices = self.df['price'].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).tolist()
        self.pos_overlap_k = max(1, int(pos_overlap_k))
        self.pos_price_delta = float(pos_price_delta)

        # Build token -> list of indices
        inv: Dict[str, List[int]] = {}
        for i, toks in enumerate(self.cats):
            for t in toks:
                if t == ROOT:
                    continue
                inv.setdefault(t, []).append(i)
        self.inv = {k: np.array(v, dtype=np.int32) for k, v in inv.items()}

        if augment:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(img_size + 20),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])

    def __len__(self):
        return len(self.paths)

    def _load(self, i: int):
        p = self.paths[i]
        with Image.open(p) as im:
            im = im.convert('RGB')
            return self.tf(im)

    def _candidates_overlap_k(self, toks: List[str], k: int) -> np.ndarray:
        """Trả về mảng indices có overlap >= k tokens với 'toks'."""
        toks = [t for t in toks if t != ROOT]
        toks = list(dict.fromkeys(toks))  # unique
        if len(toks) < k:
            return np.array([], dtype=np.int32)
        pools = []
        for comb in itertools.combinations(sorted(toks), k):
            arrs = [self.inv.get(t) for t in comb]
            if any(a is None for a in arrs):
                continue
            inter = arrs[0]
            for a in arrs[1:]:
                inter = np.intersect1d(inter, a, assume_unique=False)
                if inter.size == 0:
                    break
            if inter.size > 0:
                pools.append(inter)
        if not pools:
            return np.array([], dtype=np.int32)
        cand = np.unique(np.concatenate(pools))
        return cand

    def _sample_positive(self, i: int) -> int:
        toks = self.cats[i]
        k = self.pos_overlap_k
        cand = self._candidates_overlap_k(toks, k)
        if cand.size > 1:
            # Loại bỏ self
            cand = cand[cand != i]
        # Nếu có ràng buộc giá, lọc theo giá gần
        if self.pos_price_delta > 0 and cand.size > 0:
            pi = float(self.prices[i])
            mask = []
            for j in cand:
                pj = float(self.prices[int(j)])
                ok = price_close_log(pi, pj, self.pos_price_delta)
                mask.append(ok)
            mask = np.array(mask, dtype=bool)
            if mask.any():
                cand = cand[mask]
        # Fallback nếu không có ứng viên
        if cand.size == 0:
            # Thử giảm k nếu k > 1
            if k > 1:
                cand = self._candidates_overlap_k(toks, k=1)
                cand = cand[cand != i]
        if cand.size == 0:
            # random khác index
            j = random.randrange(0, len(self.paths))
            return j if j != i else (j + 1) % len(self.paths)
        j = int(random.choice(cand))
        return j if j != i else (j + 1) % len(self.paths)

    def __getitem__(self, i: int):
        a = self._load(i)
        j = self._sample_positive(i)
        p = self._load(j)
        # trả thêm giá để weight loss (nếu dùng)
        return a, p, i, j, float(self.prices[i]), float(self.prices[j])

# -------- Model --------
class EffB3Proj(nn.Module):
    def __init__(self, out_dim: int = 512):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features  # 1536 for B3
        self.proj = nn.Linear(feat_dim, out_dim)

    def forward(self, x):
        f = self.backbone(x)            # (B, 1536)
        z = self.proj(f)                # (B, out)
        z = F.normalize(z, dim=1)       # cosine space
        return z

# -------- InfoNCE on paired batches --------
def info_nce(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.07):
    logits = (z1 @ z2.t()) / temp  # (B,B)
    labels = torch.arange(z1.size(0), device=z1.device)
    loss1 = F.cross_entropy(logits, labels, reduction='none')  # (B,)
    loss2 = F.cross_entropy(logits.t(), labels, reduction='none')  # (B,)
    return (loss1 + loss2) * 0.5  # (B,)

def info_nce_weighted(z1: torch.Tensor, z2: torch.Tensor, w: torch.Tensor, temp: float = 0.07):
    """InfoNCE có trọng số theo cặp (i,j). w in [0,1], shape (B,)"""
    per = info_nce(z1, z2, temp=temp)  # (B,)
    # chuẩn hoá trọng số về [w_min, 1.0] để tránh triệt tiêu signal
    w = w.clamp(0, 1) * 0.5 + 0.5
    return (per * w).mean()

# -------- Validation metrics --------
@torch.no_grad()
def compute_embeddings(model: nn.Module, csv_path: str, img_size: int, batch: int, device: str, num_workers: int) -> Tuple[np.ndarray, List[List[str]], List[float]]:
    ds = PairDataset(csv_path, img_size=args.img_size, augment=False,
                     pos_overlap_k=max(1, args.pos_overlap_k), pos_price_delta=args.pos_price_delta)
    dl = DataLoader(
        ds, batch_size=batch, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    all_z = []
    pbar = tqdm(dl, desc='Val Emb', leave=False)
    for batch in pbar:  # only anchor images
        a = batch[0]  # a, p, i, j, pi, pj
        a = a.to(device, non_blocking=True)
        z = model(a)
        all_z.append(z.detach().cpu().float().numpy())
    Z = np.concatenate(all_z, axis=0)
    # cats & prices theo thứ tự anchor
    cats = [json.loads(s) for s in pd.read_csv(csv_path)['cats_tokens_json'].tolist()]
    prices = pd.read_csv(csv_path)['price'].astype(float).fillna(0.0).tolist()
    return Z, cats, prices

def categories_overlap_geK(c1: List[str], c2: List[str], k: int = 2) -> bool:
    s1, s2 = set(c1), set(c2)
    if ROOT in s1: s1.remove(ROOT)
    if ROOT in s2: s2.remove(ROOT)
    return len(s1 & s2) >= k

def price_close_pct(pi: float, pj: float, pct: float) -> bool:
    if not (pi > 0 and pj > 0):
        return False
    return (abs(pi - pj) / max(pi, pj)) <= pct

def eval_retrieval(Z: np.ndarray, cats: List[List[str]], prices: List[float],
                   topk_list=(1,10,50), overlap_k: int = 2, price_pct: float = 0.0) -> Dict[str, float]:
    # cosine sims (Z already L2 normalized)
    Zt = Z.T
    n = Z.shape[0]
    Kmax = max(topk_list) + 1
    hits_at = {k: 0 for k in topk_list}
    mrr = 0.0
    map10 = 0.0
    ndcg10_accum = 0.0

    for i in tqdm(range(n), desc='Eval NN', leave=False):
        qi = Z[i:i+1]
        sims = (qi @ Zt).ravel()
        sims[i] = -1.0  # exclude self
        idx = np.argpartition(-sims, Kmax)[:Kmax]
        idx = idx[np.argsort(-sims[idx])]

        rel = []
        for j in idx[:Kmax]:
            ok_cat = categories_overlap_geK(cats[i], cats[j], k=overlap_k)
            ok_price = True
            if price_pct > 0:
                ok_price = price_close_pct(prices[i], prices[j], pct=price_pct)
            rel.append(1 if (ok_cat and ok_price) else 0)

        # metrics
        for k in topk_list:
            if any(rel[:k]): hits_at[k] += 1
        # MRR
        rr = 0.0
        for rank, r in enumerate(rel, start=1):
            if r == 1:
                rr = 1.0 / rank; break
        mrr += rr
        # AP@10
        num_rel, prec_sum = 0, 0.0
        for rank, r in enumerate(rel[:10], start=1):
            if r == 1:
                num_rel += 1
                prec_sum += num_rel / rank
        map10 += (prec_sum / max(1, num_rel)) if num_rel > 0 else 0.0
        # nDCG@10
        def dcg(x):
            return sum(((2**xk-1)/math.log2(i+2) for i,xk in enumerate(x)))
        ndcg10_accum += dcg(rel[:10]) / max(1e-9, dcg([1]*min(10, sum(rel[:10]))))

    out = {f'Recall@{k}': hits_at[k] / n for k in topk_list}
    out['MRR'] = mrr / n
    out['mAP@10'] = map10 / n
    out['nDCG@10'] = ndcg10_accum / n
    return out

# -------- Train loop --------
def save_ckpt(path: Path, model, opt, scaler, epoch: int, best_metric: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'opt': opt.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'epoch': epoch,
        'best_metric': best_metric,
    }, path)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_csv', required=True)
    ap.add_argument('--val_csv', required=True)
    ap.add_argument('--save_dir', default='runs/effb3_cat')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--img_size', type=int, default=300)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--temp', type=float, default=0.07)
    default_workers = 0 if platform.system() == "Windows" else 6
    ap.add_argument('--num_workers', type=int, default=default_workers)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--patience', type=int, default=2, help='early stopping patience on Recall@10')
    ap.add_argument('--resume', default='')
    ap.add_argument('--log_interval', type=int, default=10, help='update tqdm postfix every N steps')

    ap.add_argument('--pos_overlap_k', type=int, default=2, help='số token overlap để coi là positive (train sampling)')
    ap.add_argument('--pos_price_delta', type=float, default=0.0, help='ràng buộc giá khi sample positive (|log p_i - log p_j| <= delta); 0 = tắt')
    ap.add_argument('--price_weight', type=float, default=0.0, help='trọng số ảnh hưởng giá trong loss; 0 = tắt')
    ap.add_argument('--price_sigma', type=float, default=0.2, help='độ rộng Gaussian trong log-price cho weight (0.2 ≈ ±22%)')

    ap.add_argument('--eval_overlap_k', type=int, default=2, help='overlap tokens khi tính relevance ở eval')
    ap.add_argument('--eval_price_pct', type=float, default=0.0, help='ràng buộc giá ở eval: ±pct; 0 = tắt')
    return ap.parse_args()

def main():
    global args
    args = parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = PairDataset(args.train_csv, img_size=args.img_size, augment=True,
                     pos_overlap_k=args.pos_overlap_k, pos_price_delta=args.pos_price_delta)
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )

    model = EffB3Proj(out_dim=512).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    try:
        scaler = torch.amp.GradScaler(device_type='cuda', enabled=torch.cuda.is_available())
    except TypeError:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler(enabled=torch.cuda.is_available())

    start_epoch = 0
    best_metric = -1.0
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / 'train_log.jsonl'

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        if ckpt.get('scaler') and scaler is not None:
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_metric = ckpt.get('best_metric', -1.0)
        print(f"Resumed from {args.resume} @ epoch {start_epoch}")

    # --- train epochs ---
    no_improve = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        ema_loss = None
        n_seen = 0
        t0 = time.time()

        pbar = tqdm(dl, desc=f'Epoch {epoch+1}/{args.epochs}', dynamic_ncols=True)
        for step, batch in enumerate(pbar, start=1):
            a, p, _, _, pa, pp = batch  # pa, pp: price anchors/positives
            a = a.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)
            pa = torch.tensor(pa, dtype=torch.float32, device=device) if not torch.is_tensor(pa) else pa.to(device)
            pp = torch.tensor(pp, dtype=torch.float32, device=device) if not torch.is_tensor(pp) else pp.to(device)

            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                z1 = model(a)
                z2 = model(p)
                if args.price_weight > 0.0:
                    w = price_affinity_torch(pa, pp, sigma=args.price_sigma)  # (B,)
                    per_loss = info_nce(z1, z2, temp=args.temp)               # (B,)
                    # kết hợp: (1 - λ)*CE + λ*weighted
                    base = per_loss.mean()
                    weighted = (per_loss * (w.clamp(0,1)*0.5 + 0.5)).mean()
                    loss = (1 - args.price_weight) * base + args.price_weight * weighted
                else:
                    loss = info_nce(z1, z2, temp=args.temp).mean()

            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward(); opt.step()

            # stats
            loss_val = float(loss.detach().cpu())
            epoch_loss += loss_val
            ema_loss = loss_val if ema_loss is None else (0.9 * ema_loss + 0.1 * loss_val)
            n_seen += a.size(0)

            if step % max(1, args.log_interval) == 0:
                elapsed = max(1e-6, time.time() - t0)
                samples_per_s = n_seen / elapsed
                pbar.set_postfix({
                    'loss': f"{loss_val:.4f}",
                    'ema': f"{ema_loss:.4f}",
                    'lr': f"{get_lr(opt):.2e}",
                    'spd': f"{samples_per_s:.0f}/s",
                    'mem': gpu_mem()
                })

        epoch_loss /= max(1, len(dl))

        # --- validate ---
        model.eval()
        val_workers = max(0, min(2, args.num_workers))  # nhẹ cho Windows
        Z, cats, prices = compute_embeddings(model, args.val_csv, args.img_size,
                                             batch=max(64, args.batch//2), device=device, num_workers=val_workers)
        metrics = eval_retrieval(
            Z, cats, prices,
            topk_list=(1,10,50),
            overlap_k=args.eval_overlap_k,
            price_pct=args.eval_price_pct
        )
        metrics['Loss'] = epoch_loss
        metrics['epoch'] = epoch
        metrics['time_min'] = (time.time()-t0)/60

        with open(log_path, 'a', encoding='utf-8') as w:
            w.write(json.dumps(metrics, ensure_ascii=False) + '\n')

        # save last
        save_ckpt(save_dir / 'ckpt_last.pth', model, opt, scaler, epoch, best_metric)

        cur = metrics['Recall@10']
        print(f"\n[Epoch {epoch+1}] Loss={epoch_loss:.4f} | R@1={metrics['Recall@1']:.4f} "
              f"| R@10={metrics['Recall@10']:.4f} | mAP@10={metrics['mAP@10']:.4f} | MRR={metrics['MRR']:.4f}")

        if cur > best_metric:
            best_metric = cur
            save_ckpt(save_dir / 'ckpt_best.pth', model, opt, scaler, epoch, best_metric)
            no_improve = 0
            print(f"✅ New best Recall@10={cur:.4f} @ epoch {epoch}")
        else:
            no_improve += 1
            print(f"No improvement ({no_improve}/{args.patience}). Best Recall@10={best_metric:.4f}")
            if no_improve >= args.patience:
                print('Early stopping triggered.')
                break

    print('Done. Checkpoints at', save_dir)

if __name__ == '__main__':
    main()
