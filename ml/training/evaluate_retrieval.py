import argparse
import os
import random

import numpy as np
import pandas as pd
import faiss
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def load_npy(path, desc):
    print(f"[INFO] Loading {desc} from: {path}")
    arr = np.load(path, allow_pickle=True)
    print(f"[INFO] {desc} shape/len: {getattr(arr, 'shape', len(arr))}")
    return arr


def load_faiss_index(index_path):
    print(f"[INFO] Loading FAISS index from: {index_path}")
    index = faiss.read_index(index_path)
    print(f"[INFO] Index loaded. ntotal = {index.ntotal}, dim = {index.d}")
    return index


def load_bboxes_from_metadata(metadata_path):
    print(f"[INFO] Loading bboxes from metadata: {metadata_path}")
    df = pd.read_csv(metadata_path)
    if "split" not in df.columns or not all(c in df.columns for c in ["x1", "y1", "x2", "y2"]):
        raise ValueError("Metadata must have columns: split, x1, y1, x2, y2")
    query_df = df[df["split"] == "query"].reset_index(drop=True)
    gallery_df = df[df["split"] == "gallery"].reset_index(drop=True)
    query_bboxes = [
        (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
        for _, row in query_df.iterrows()
    ]
    gallery_bboxes = [
        (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
        for _, row in gallery_df.iterrows()
    ]
    print(f"[INFO] Loaded {len(query_bboxes)} query bboxes, {len(gallery_bboxes)} gallery bboxes")
    return query_bboxes, gallery_bboxes


def compute_ap_at_k(relevant, k, total_relevant):
    k = min(k, len(relevant))
    relevant = relevant[:k]

    precisions = []
    num_rel_so_far = 0

    for i, is_rel in enumerate(relevant, start=1):
        if is_rel:
            num_rel_so_far += 1
            precisions.append(num_rel_so_far / i)

    if total_relevant == 0:
        return 0.0

    return float(sum(precisions) / total_relevant)
    

def evaluate_retrieval(index, query_feats, gallery_labels, query_labels, topk):
    num_queries = query_feats.shape[0]
    sims, idxs = index.search(query_feats, topk)

    all_ap = []
    all_recall = []

    for qi in range(num_queries):
        q_label = query_labels[qi]
        retrieved_idx = idxs[qi]  # (topk,)

        total_relevant = np.sum(gallery_labels == q_label)

        if total_relevant == 0:
            continue

        is_relevant = np.zeros_like(retrieved_idx, dtype=bool)

        valid_mask = (retrieved_idx != -1)
        valid_idx = retrieved_idx[valid_mask]

        if np.any(valid_mask):
            retrieved_labels = gallery_labels[valid_idx]
            is_relevant[valid_mask] = (retrieved_labels == q_label)

        ap = compute_ap_at_k(is_relevant, topk, total_relevant)
        all_ap.append(ap)

        recall_at_k = np.sum(is_relevant) / float(total_relevant)
        all_recall.append(recall_at_k)

    mAP = float(np.mean(all_ap)) if len(all_ap) > 0 else 0.0
    mean_recall = float(np.mean(all_recall)) if len(all_recall) > 0 else 0.0

    return mAP, mean_recall


def evaluate_retrieval_multi_topk(
    index, query_feats, gallery_labels, query_labels,
    recall_ks=(1, 5, 10), map_ks=(10, 50),
):
    max_k = max(max(recall_ks), max(map_ks))
    num_queries = query_feats.shape[0]
    sims, idxs = index.search(query_feats, max_k)

    recall_results = {k: [] for k in recall_ks}
    map_results = {k: [] for k in map_ks}

    for qi in range(num_queries):
        q_label = query_labels[qi]
        retrieved_idx = idxs[qi]
        total_relevant = np.sum(gallery_labels == q_label)

        if total_relevant == 0:
            continue

        is_relevant = np.zeros(len(retrieved_idx), dtype=bool)
        valid_mask = (retrieved_idx != -1)
        valid_idx = retrieved_idx[valid_mask]
        if np.any(valid_mask):
            retrieved_labels = gallery_labels[valid_idx]
            is_relevant[valid_mask] = (retrieved_labels == q_label)

        for k in recall_ks:
            recall_at_k = np.sum(is_relevant[:k]) / float(total_relevant)
            recall_results[k].append(recall_at_k)
        for k in map_ks:
            ap = compute_ap_at_k(is_relevant, k, total_relevant)
            map_results[k].append(ap)

    out = {}
    for k in recall_ks:
        out[f"Recall@{k}"] = float(np.mean(recall_results[k])) if recall_results[k] else 0.0
    for k in map_ks:
        out[f"mAP@{k}"] = float(np.mean(map_results[k])) if map_results[k] else 0.0
    return out


def _load_image(path, image_root, fallback_color=(128, 128, 128)):
    full_path = os.path.join(image_root, str(path)) if image_root else str(path)
    try:
        return Image.open(full_path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Failed to open image {full_path}: {e}")
        return Image.new("RGB", (256, 256), color=fallback_color)


def _crop_bbox(pil_img, bbox):
    x1, y1, x2, y2 = bbox
    return pil_img.crop((x1, y1, x2, y2))


def visualize_results(
    index,
    query_feats,
    gallery_paths,
    query_paths,
    out_dir,
    num_queries_vis=5,
    topk=5,
    image_root="",
    query_bboxes=None,
    gallery_bboxes=None,
):
    os.makedirs(out_dir, exist_ok=True)
    num_queries = query_feats.shape[0]
    use_bbox = query_bboxes is not None and gallery_bboxes is not None


    sampled_q_indices = random.sample(range(num_queries), k=min(num_queries_vis, num_queries))
    print(f"[INFO] Visualizing {len(sampled_q_indices)} random queries: {sampled_q_indices}")

    sims, idxs = index.search(query_feats[sampled_q_indices], topk)

    cols = topk + 1
    rows = 2 if use_bbox else 1
    fig_h = 3 * rows

    for i, qi in enumerate(sampled_q_indices):
        q_path = query_paths[qi]
        q_img = _load_image(q_path, image_root, (128, 128, 128))
        q_bbox = query_bboxes[qi] if use_bbox else None

        plt.figure(figsize=(3 * cols, fig_h))

        ax0 = plt.subplot(rows, cols, 1)
        ax0.imshow(q_img)
        if q_bbox is not None:
            x1, y1, x2, y2 = q_bbox
            ax0.add_patch(
                Rectangle((x1, y2), x2 - x1, y1 - y2, linewidth=2, edgecolor="lime", facecolor="none")
            )
        ax0.axis("off")
        ax0.set_title("Query (full)" if use_bbox else "Query")

        if use_bbox:
            ax1 = plt.subplot(rows, cols, cols + 1)
            ax1.imshow(_crop_bbox(q_img, q_bbox))
            ax1.axis("off")
            ax1.set_title("Query (crop)")

        for rank in range(topk):
            g_idx = idxs[i, rank]
            col = rank + 2  # 1-based subplot index

            if g_idx == -1:
                placeholder = Image.new("RGB", (256, 256), color=(200, 200, 200))
                ax_full = plt.subplot(rows, cols, col)
                ax_full.imshow(placeholder)
                ax_full.axis("off")
                ax_full.set_title(f"Top {rank+1} (n/a)")
                if use_bbox:
                    ax_crop = plt.subplot(rows, cols, cols + col)
                    ax_crop.imshow(placeholder)
                    ax_crop.axis("off")
                    ax_crop.set_title(f"Top {rank+1} crop (n/a)")
                continue

            g_path = gallery_paths[g_idx]
            g_img = _load_image(g_path, image_root, (200, 200, 200))
            g_bbox = gallery_bboxes[g_idx] if use_bbox else None

            ax_full = plt.subplot(rows, cols, col)
            ax_full.imshow(g_img)
            if g_bbox is not None:
                x1, y1, x2, y2 = g_bbox
                ax_full.add_patch(
                    Rectangle((x1, y2), x2 - x1, y1 - y2, linewidth=2, edgecolor="lime", facecolor="none")
                )
            ax_full.axis("off")
            ax_full.set_title(f"Top {rank+1} (full)" if use_bbox else f"Top {rank+1}")

            if use_bbox:
                ax_crop = plt.subplot(rows, cols, cols + col)
                ax_crop.imshow(_crop_bbox(g_img, g_bbox))
                ax_crop.axis("off")
                ax_crop.set_title(f"Top {rank+1} (crop)")

        out_path = os.path.join(out_dir, f"query_{qi}.png")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"[INFO] Saved visualization: {out_path}")


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    index = load_faiss_index(args.index_path)
    query_feats = load_npy(args.query_feats, "query_feats")
    gallery_paths = load_npy(args.gallery_paths, "gallery_paths")
    query_paths = load_npy(args.query_paths, "query_paths")
    gallery_labels = load_npy(args.gallery_labels, "gallery_labels")
    query_labels = load_npy(args.query_labels, "query_labels")

    if index.d != query_feats.shape[1]:
        raise ValueError(f"Dim mismatch: index dim={index.d}, query dim={query_feats.shape[1]}")

    if len(gallery_paths) != index.ntotal:
        print(
            f"[WARN] len(gallery_paths)={len(gallery_paths)} != index.ntotal={index.ntotal}. "
            "There may be a mismatch between the paths and the indices."
        )

    print("[INFO] Evaluating retrieval metrics...")
    mAP, recall = evaluate_retrieval(
        index=index,
        query_feats=query_feats,
        gallery_labels=gallery_labels,
        query_labels=query_labels,
        topk=args.topk,
    )
    print(f"[METRIC] mAP@{args.topk}      = {mAP:.4f}")
    print(f"[METRIC] Recall@{args.topk}   = {recall:.4f}")

    query_bboxes, gallery_bboxes = None, None
    if getattr(args, "metadata", None):
        query_bboxes, gallery_bboxes = load_bboxes_from_metadata(args.metadata)
        if len(query_bboxes) != len(query_paths) or len(gallery_bboxes) != len(gallery_paths):
            print(
                "[WARN] Bbox count mismatch with paths. Disabling bbox visualization."
            )
            query_bboxes, gallery_bboxes = None, None

    if args.num_vis_queries > 0:
        print("[INFO] Visualizing some retrieval results...")
        visualize_results(
            index=index,
            query_feats=query_feats,
            gallery_paths=gallery_paths,
            query_paths=query_paths,
            out_dir=args.vis_out_dir,
            num_queries_vis=args.num_vis_queries,
            topk=args.topk,
            image_root=args.image_root,
            query_bboxes=query_bboxes,
            gallery_bboxes=gallery_bboxes,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval (mAP, Recall@K) and visualize top-k results."
    )
    parser.add_argument(
        "--index_path",
        type=str,
        required=True,
        help="Path to FAISS index (e.g. indices/dinov2_frozen/faiss_gallery_index.ip)",
    )
    parser.add_argument(
        "--query_feats",
        type=str,
        required=True,
        help="Path to query features .npy",
    )
    parser.add_argument(
        "--gallery_paths",
        type=str,
        required=True,
        help="Path to gallery image paths .npy (len = #gallery)",
    )
    parser.add_argument(
        "--query_paths",
        type=str,
        required=True,
        help="Path to query image paths .npy (len = #queries)",
    )
    parser.add_argument(
        "--gallery_labels",
        type=str,
        required=True,
        help="Path to gallery item_ids .npy (len = #gallery)",
    )
    parser.add_argument(
        "--query_labels",
        type=str,
        required=True,
        help="Path to query item_ids .npy (len = #queries)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Top-K for metrics and visualization.",
    )
    parser.add_argument(
        "--num_vis_queries",
        type=int,
        default=5,
        help="Number of random queries to visualize.",
    )
    parser.add_argument(
        "--vis_out_dir",
        type=str,
        default="retrieval_vis",
        help="Directory to save visualization images.",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="",
        help="Optional root dir to prepend to image paths.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Path to metadata CSV (with split, x1,y1,x2,y2). If set, visualization shows full image + bbox crop.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling queries.",
    )

    args = parser.parse_args()
    main(args)

