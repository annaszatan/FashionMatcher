"""
The purpose of this file is to search the FAISS index for matching gallery items for a given query item
Usage example:
python search_faiss_index.py \
    --index_path faiss_gallery_index.ip \
    --query_path query_embeddings.npy \
    --topk 5 \
    --gallery_paths gallery_paths.txt
"""

import argparse
import numpy as np
import faiss
import os


def load_query_embeddings(path):
    print(f"[INFO] Loading query embeddings from: {path}")
    queries = np.load(path).astype("float32")
    print(f"[INFO] Loaded query embeddings with shape: {queries.shape}")
    return queries


def load_gallery_paths(path):
    if path is None:
        return None

    print(f"[INFO] Loading gallery paths from: {path}")
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        paths = np.load(path)
        paths = paths.tolist()
    else:
        with open(path, "r", encoding="utf-8") as f:
            paths = [line.strip() for line in f if line.strip()]

    print(f"[INFO] Loaded {len(paths)} gallery paths")
    return paths


def search_index(index_path, query_embeddings, topk, gallery_paths=None):
    print(f"[INFO] Loading FAISS index from: {index_path}")
    index = faiss.read_index(index_path)
    print(f"[INFO] Index loaded. Total items in index: {index.ntotal}")

    d_index = index.d
    d_query = query_embeddings.shape[1]
    if d_index != d_query:
        raise ValueError(
            f"Dimension mismatch: index dim={d_index}, query dim={d_query}"
        )

    print(f"[INFO] Searching top-{topk} nearest neighbors...")
    sims, idxs = index.search(query_embeddings, topk)
    print("[INFO] Search complete.")

    num_queries = query_embeddings.shape[0]
    print(f"[INFO] Number of queries: {num_queries}")

    for qi in range(num_queries):
        print("=" * 60)
        print(f"[RESULT] Query {qi} (top-{topk})")
        for rank, (g_idx, sim) in enumerate(zip(idxs[qi], sims[qi]), start=1):
            if g_idx == -1:
                continue
            if gallery_paths is not None and 0 <= g_idx < len(gallery_paths):
                g_info = gallery_paths[g_idx]
            else:
                g_info = f"gallery_index={g_idx}"

            print(f"  Rank {rank:2d}: {g_info}  (cos_sim={sim:.4f})")


def main(args):
    query_embeddings = load_query_embeddings(args.query_path)

    gallery_paths = load_gallery_paths(args.gallery_paths)

    search_index(
        index_path=args.index_path,
        query_embeddings=query_embeddings,
        topk=args.topk,
        gallery_paths=gallery_paths,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search FAISS index with query embeddings (cosine similarity via inner product)."
    )
    parser.add_argument(
        "--index_path",
        type=str,
        required=True,
        help="Path to FAISS index file (e.g., faiss_gallery_index.ip)",
    )
    parser.add_argument(
        "--query_path",
        type=str,
        required=True,
        help="Path to query embeddings .npy file",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of nearest neighbors to retrieve per query",
    )
    parser.add_argument(
        "--gallery_paths",
        type=str,
        default=None,
        help=(
            "Optional: path to gallery image paths file "
            "(.npy with list/array of strings, or .txt with one path per line)"
        ),
    )

    args = parser.parse_args()
    main(args)
