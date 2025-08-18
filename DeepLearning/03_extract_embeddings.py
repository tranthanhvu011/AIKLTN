# 03_extract_embeddings.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch, torch.nn.functional as F
from torchvision import transforms
import timm

class EffB3Proj(torch.nn.Module):
    def __init__(self, out_dim: int = 512):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b3', pretrained=False, num_classes=0)
        self.proj = torch.nn.Linear(self.backbone.num_features, out_dim)
    def forward(self, x):
        f = self.backbone(x)
        z = self.proj(f)
        return F.normalize(z, dim=1)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out_dir', default='emb_out')
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--img_size', type=int, default=300)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    tf = transforms.Compose([
        transforms.Resize(args.img_size + 20),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EffB3Proj(out_dim=512).to(device)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model']); model.eval()

    paths = df['image_path'].tolist(); asins = df['asin'].tolist()
    emb = []
    batch_imgs = []

    def load_img(p):
        from PIL import Image
        with Image.open(p) as im:
            return tf(im.convert('RGB'))

    for i, p in enumerate(tqdm(paths, desc='Embed')):
        batch_imgs.append(load_img(p))
        if len(batch_imgs) == args.batch or i == len(paths)-1:
            x = torch.stack(batch_imgs).to(device)
            z = model(x).detach().cpu().float().numpy()
            emb.append(z)
            batch_imgs.clear()

    E = np.concatenate(emb, axis=0)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'embeddings.npy', E)
    pd.DataFrame({'asin': asins, 'image_path': paths}).to_csv(out / 'mapping.csv', index=False)
    print('Saved:')
    print(' -', out / 'embeddings.npy')
    print(' -', out / 'mapping.csv')

if __name__ == '__main__':
    main()