# 04_build_faiss_index.py
import argparse, numpy as np, faiss
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--emb', required=True, help='embeddings.npy')
    ap.add_argument('--out', required=True, help='index.faiss')
    ap.add_argument('--factory', default='HNSW32', help='FlatIP | HNSW32 | IVF4096,Flat | IVF8192,PQ64 ...')
    args = ap.parse_args()

    E = np.load(args.emb).astype('float32')
    # Ensure L2-normalized for cosine/IP
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    E = E / norms

    d = E.shape[1]
    if args.factory.lower() in ('flat', 'flatip'):
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.index_factory(d, args.factory, faiss.METRIC_INNER_PRODUCT)
        if not index.is_trained:
            index.train(E)
    index.add(E)

    faiss.write_index(index, args.out)
    print('Saved:', args.out, '| ntotal =', index.ntotal)

if __name__ == '__main__':
    main()