# 05_search_image.py
import argparse, numpy as np, pandas as pd, faiss
from PIL import Image
import torch, torch.nn.functional as F
from torchvision import transforms
import timm

class EffB3Proj(torch.nn.Module):
    def __init__(self, out_dim: int = 512):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b3', pretrained=False, num_classes=0)
        self.proj = torch.nn.Linear(self.backbone.num_features, out_dim)
    def forward(self, x):
        z = self.proj(self.backbone(x))
        return F.normalize(z, dim=1)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--index', required=True)
    ap.add_argument('--mapping_csv', required=True)
    ap.add_argument('--query_img', required=True)
    ap.add_argument('--topk', type=int, default=10)
    ap.add_argument('--img_size', type=int, default=300)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EffB3Proj(512).to(device)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model']); model.eval()

    tf = transforms.Compose([
        transforms.Resize(args.img_size + 20),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    with Image.open(args.query_img) as im:
        x = tf(im.convert('RGB')).unsqueeze(0).to(device)
    z = model(x).cpu().numpy().astype('float32')

    index = faiss.read_index(args.index)
    D, I = index.search(z, args.topk)

    mapping = pd.read_csv(args.mapping_csv)
    for rank, idx in enumerate(I[0], start=1):
        row = mapping.iloc[idx]
        print(f"#{rank}  asin={row['asin']}  path={row['image_path']}  score={D[0][rank-1]:.4f}")

if __name__ == '__main__':
    main()