import argparse
import numpy as np
import faiss
import os

def load_embeddings(path):
    print(f"[INFO] Loading gallery embeddings from: {path}")
    embeddings = np.load(path).astype("float32")
    print(f"[INFO] Loaded embeddings with shape: {embeddings.shape}")
    return embeddings

def build_faiss_index(embeddings, save_path):
    d = embeddings.shape[1]  # embedding dimension
    print(f"[INFO] Building FAISS IndexFlatIP (dim={d})")

    index = faiss.IndexFlatIP(d)

    print("[INFO] Adding embeddings to index...")
    index.add(embeddings)

    print(f"[INFO] Index built. Total items: {index.ntotal}")
    print(f"[INFO] Saving index to {save_path}")
    faiss.write_index(index, save_path)
    print("[INFO] Done.")

def main(args):
    gallery = load_embeddings(args.emb_path)

    # gallery_norm = l2_normalize(gallery)

    build_faiss_index(gallery, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a FAISS cosine similarity index from gallery embeddings."
    )
    parser.add_argument(
        "--emb_path",
        type=str,
        required=True,
        help="Path to gallery embeddings .npy file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="faiss_gallery_index.ip",
        help="Output FAISS index filename"
    )

    args = parser.parse_args()
    main(args)
