import torch
import torchvision.io as io
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

DEF_SIZE = 300

def resize_gpu(src_path: Path, dst_path: Path, size: int = DEF_SIZE):
    try:
        img = io.read_image(str(src_path)).float() / 255.0  # (C,H,W)
        img = img.unsqueeze(0).to('cuda')  # batch=1
        _, _, h, w = img.shape
        scale = size / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = F.interpolate(img, size=(new_h, new_w), mode='bicubic', align_corners=False)
        # center crop
        top = (new_h - size) // 2
        left = (new_w - size) // 2
        img = img[:, :, top:top+size, left:left+size]
        img = (img.clamp(0, 1) * 255).byte().cpu().squeeze(0)
        io.write_jpeg(img, str(dst_path), quality=92)
    except:
        pass

src = Path("./output/images_topk")
dst = Path("./output/images_300")
files = [p for p in src.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}]

for f in tqdm(files):
    out = (dst / f.relative_to(src)).with_suffix(".jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    resize_gpu(f, out)
