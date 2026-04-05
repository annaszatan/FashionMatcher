"""
Evaluate end-to-end FashionRetrievalPipeline behavior on a query set.

Using --metadata_csv, this script auto-builds query rows by:
- choosing one deterministic gallery item_id as gt_item_id

HOW TO USE:
```
python scripts/evaluate_pipeline.py --metdata_csv path/to/metadata.csv --output_dir path/to/save/results

Outputs:
- summary_metrics.json
- per_query_results.csv
- good_examples.csv / bad_examples.csv
- good_examples_grid.png / bad_examples_grid.png (if matplotlib is available)
"""

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image

try:
	from tqdm import tqdm
except ImportError:
	tqdm = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from ml.pipeline import FashionRetrievalPipeline


@dataclass
class QueryEvalResult:
	image_path: str
	gt_item_id: str
	num_crops: int
	top1_item_id: str
	top1_score: float
	hit_at_1: int
	hit_at_5: int
	hit_at_10: int


def build_queries_from_metadata(
	metadata_df: pd.DataFrame,
	image_col: str,
	gt_col: str,
	query_split: str = "query",
	gallery_split: str = "gallery",
	match_on: str = "pair_id",
) -> pd.DataFrame:

	# Check the query and gallery splits exist in the metadata file
	queries = metadata_df[metadata_df["split"] == query_split].copy()
	gallery = metadata_df[metadata_df["split"] == gallery_split].copy()
	if queries.empty:
		raise ValueError(f"No query rows found in metadata (split == '{query_split}').")
	if gallery.empty:
		raise ValueError(f"No gallery rows found in metadata (split == '{gallery_split}').")

	# For each query, find gallery candidates that share the same pair_id or item_id (can further filter by category)
	use_category = "category" in metadata_df.columns
	rows = []
	for _, q in queries.iterrows():
		pair_id = q["pair_id"]
		q_cat = q["category"] if use_category else None

		# Find gallery candidates with same pair_id
		cands = gallery[gallery["pair_id"] == pair_id]
		if use_category:
			cands_cat = cands[cands["category"] == q_cat]
			if not cands_cat.empty:
				cands = cands_cat

		if cands.empty:
			continue
		
		# For this query, assign gt label based on match_on criteria and add to rows for evaluation
		if match_on == "pair_id":
			rows.append({image_col: q["image_path"], gt_col: str(pair_id)})
		# If matching on item_id, we need to choose one deterministic gallery item_id as the gt label for this query
		elif match_on == "item_id":
			# Deterministic single-label target for current evaluator
			cands = cands.sort_values(by=["item_id", "image_path"], kind="stable")
			rows.append({image_col: q["image_path"], gt_col: str(cands.iloc[0]["item_id"])})
		else:
			raise ValueError(f"Unsupported match_on: {match_on}")

	if not rows:
		raise ValueError("No evaluable query rows could be built from metadata CSV.")

	return pd.DataFrame(rows)

# Allow for absolute and relative paths
def resolve_image_path(image_path: str, image_root: str) -> str:
	if os.path.isabs(image_path):
		return image_path
	return os.path.normpath(os.path.join(image_root, image_path))


def get_projected_features_dir(embedding_folder: str) -> str:
	validation_dir = os.path.join(PROJECT_ROOT, "ml", "features", embedding_folder)
	if not os.path.isdir(validation_dir):
		raise FileNotFoundError(f"Validation features directory not found: {validation_dir}")
	return validation_dir


def merge_crop_results(crop_results: List[List[Dict]], topk: int, strategy: str = "max") -> List[Dict]:
	"""Merge retrieval lists across crops by item using max/avg/sum aggregation."""
	# Sort crop results by descending and keep track of each unique item_id's best score
	best_by_key: Dict[str, Dict] = {}
	scores_by_key: Dict[str, List[float]] = {}
	# Loop through crop results
	for results in crop_results:
		# Loop through retrieval results for each crop
		for rec in results:
			key = str(rec.get("item_id") or rec.get("path") or rec.get("gallery_index"))
			score = float(rec.get("score", -1e9))
			# Add all scores for this key to the scores_by_key dict for later aggregation
			scores_by_key.setdefault(key, []).append(score)
			# If key not in best by key dict or the score is better than existing, update best record for this key
			if key not in best_by_key or score > float(best_by_key[key].get("score", -1e9)):
				best_by_key[key] = dict(rec)

	# For each unique item_id key, aggregate score across crops using specificed strategy
	for key, rec in best_by_key.items():
		# Get all scores for this key across crops
		scores = scores_by_key.get(key, [float(rec.get("score", -1e9))])
		# Return the best score
		if strategy == "max":
			rec["score"] = max(scores)
		# Return the average score
		elif strategy == "avg":
			rec["score"] = float(sum(scores) / len(scores))
		# Return the sum of scores
		elif strategy == "sum":
			rec["score"] = float(sum(scores))
		else:
			raise ValueError(f"Unsupported crop_fusion strategy: {strategy}")

	# Sort the merged results by descending score and return the topk records
	merged = sorted(best_by_key.values(), key=lambda x: float(x.get("score", -1e9)), reverse=True)
	return merged[:topk]

# Get pair id from item_id
def pair_id_from_item_id(item_id: str) -> str:
	parts = str(item_id).split("_", 1)
	return parts[0] if parts else ""

# Calculate the hit@k metric for each query based on the ranked retrieval results
def hit_at_k(ranked: List[Dict], gt_label: str, k: int, match_on: str) -> int:
	# Get the top k results
	topk = ranked[:k]
	# For each record in top k results, check if it matches the ground truth label based on the defined match criteria
	for rec in topk:
		pred_item_id = str(rec.get("item_id", ""))
		if match_on == "item_id":
			if pred_item_id == str(gt_label):
				return 1
		elif match_on == "pair_id":
			if pair_id_from_item_id(pred_item_id) == str(gt_label):
				return 1
		else:
			raise ValueError(f"Unsupported match_on: {match_on}")
	return 0

# Calculate the precision@k metric
def precision_at_k(hit: int, k: int) -> float:
	# Single-relevant-item assumption per query.
	return (1.0 / float(k)) if hit else 0.0

# Make directory if it doesnt exist
def safe_makedirs(path: str):
	os.makedirs(path, exist_ok=True)


def unique_by_image_path(rows: List[QueryEvalResult]) -> List[QueryEvalResult]:
	"""Keep first occurrence per image path while preserving ranked order."""
	seen = set()
	unique_rows: List[QueryEvalResult] = []
	for row in rows:
		key = str(row.image_path)
		if key in seen:
			continue
		seen.add(key)
		unique_rows.append(row)
	return unique_rows

# This function creates a grid of images with their evaluation results
def build_grid(rows: List[QueryEvalResult], output_path: str, title: str):
	try:
		import matplotlib.pyplot as plt
		from PIL import Image
	except Exception:
		return False

	if not rows:
		return False

	cols = 5
	n = len(rows)
	nrows = int(math.ceil(n / cols))
	fig, axes = plt.subplots(nrows, cols, figsize=(4 * cols, 3.8 * nrows))
	if nrows == 1:
		axes = [axes]

	for i in range(nrows * cols):
		r = i // cols
		c = i % cols
		ax = axes[r][c] if nrows > 1 else axes[c]
		ax.axis("off")

		if i >= n:
			continue

		row = rows[i]
		try:
			img = Image.open(row.image_path).convert("RGB")
			ax.imshow(img)
		except Exception:
			ax.text(0.5, 0.5, "Image load failed", ha="center", va="center")

		ax.set_title(
			f"GT: {row.gt_item_id}\n"
			f"Top1: {row.top1_item_id}\n"
			f"H@1/5/10: {row.hit_at_1}/{row.hit_at_5}/{row.hit_at_10}",
			fontsize=9,
		)

	fig.suptitle(title, fontsize=14)
	fig.tight_layout()
	fig.savefig(output_path, dpi=150)
	plt.close(fig)
	return True


def evaluate(args):
	# Create output dir if not exists
	safe_makedirs(args.output_dir)

	# Create the queries from the metadata CSV file
	meta_df = pd.read_csv(args.metadata_csv)
	df = build_queries_from_metadata(
		metadata_df=meta_df,
		image_col=args.image_col,
		gt_col=args.gt_col,
		query_split=args.metadata_query_split,
		gallery_split=args.metadata_gallery_split,
		match_on=args.match_on,
	)
	print(
		f"Built {len(df)} query rows from metadata CSV: {args.metadata_csv} "
		f"(query_split={args.metadata_query_split}, gallery_split={args.metadata_gallery_split})"
	)

	print("Initializing pipeline components...")

	# Initialize full pipeline
	pipeline = FashionRetrievalPipeline(
		localization_folder=args.localization_folder,
		embedding_folder=args.embedding_folder,
		embedding_config=args.embedding_config,
		is_demo=False
	)

	# If not running end-to-end eval, run diagnostics for localization and/or retrieval components in isolation
	if args.diagnostic_mode != "end_to_end":
		return diagnose(args, pipeline, meta_df)

	results: List[QueryEvalResult] = []

	rows_iter = df.iterrows()
	if tqdm is not None:
		rows_iter = tqdm(rows_iter, total=len(df), desc="Evaluating queries", unit="img")

	# Loop through all query rows and run through the pipeline to get retrieval results
	for _, row in rows_iter:
		rel_path = str(row[args.image_col])
		gt_label = str(row[args.gt_col])
		image_path = resolve_image_path(rel_path, args.image_root)

		# If image path doesn't exist, count as a miss and continue to next query
		if not os.path.isfile(image_path):
			results.append(
				QueryEvalResult(
					image_path=image_path,
					gt_item_id=gt_label,
					num_crops=0,
					top1_item_id="",
					top1_score=0.0,
					hit_at_1=0,
					hit_at_5=0,
					hit_at_10=0,
				)
			)
			continue
		
		# Get list of crop paths and retrieval results for each crop from the pipeline
		crop_results = pipeline.run(image_path)

		# If no crops were detected, count as a miss and continue to next query
		if not crop_results:
			results.append(
				QueryEvalResult(
					image_path=image_path,
					gt_item_id=gt_label,
					num_crops=0,
					top1_item_id="",
					top1_score=0.0,
					hit_at_1=0,
					hit_at_5=0,
					hit_at_10=0,
				)
			)
			continue
		
		# Merge retrieval results across crops 
		ranked = merge_crop_results(crop_results, topk=10, strategy=args.crop_fusion)

		# Aggregate results
		top1_item_id = ""
		top1_score = 0.0
		if ranked:
			top1_item_id = str(ranked[0].get("item_id", ""))
			top1_score = float(ranked[0].get("score", 0.0))

		h1 = hit_at_k(ranked, gt_label, 1, args.match_on)
		h5 = hit_at_k(ranked, gt_label, 5, args.match_on)
		h10 = hit_at_k(ranked, gt_label, 10, args.match_on)

		results.append(
			QueryEvalResult(
				image_path=image_path,
				gt_item_id=gt_label,
				num_crops=len(crop_results),
				top1_item_id=top1_item_id,
				top1_score=top1_score,
				hit_at_1=h1,
				hit_at_5=h5,
				hit_at_10=h10,
			)
		)

	if not results:
		raise RuntimeError("No rows were evaluated.")
	
	# Calculate the metrics (Recall & Precision)
	n = float(len(results))
	recall_1 = sum(r.hit_at_1 for r in results) / n
	recall_5 = sum(r.hit_at_5 for r in results) / n
	recall_10 = sum(r.hit_at_10 for r in results) / n

	precision_1 = sum(precision_at_k(r.hit_at_1, 1) for r in results) / n
	precision_5 = sum(precision_at_k(r.hit_at_5, 5) for r in results) / n
	precision_10 = sum(precision_at_k(r.hit_at_10, 10) for r in results) / n
	
	# Compile summary
	summary = {
		"num_queries": int(n),
		"match_on": args.match_on,
		"precision@1": precision_1,
		"precision@5": precision_5,
		"precision@10": precision_10,
		"recall@1": recall_1,
		"recall@5": recall_5,
		"recall@10": recall_10,
	}

	with open(os.path.join(args.output_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	out_df = pd.DataFrame([r.__dict__ for r in results])
	out_df.to_csv(os.path.join(args.output_dir, "per_query_results.csv"), index=False)

	# Compile good and bad examples based on metrics, and save CSVs and grids for each.
	# Best: prefer hit@10, then hit@5, then hit@1, then higher score.
	good_sorted = sorted(
		results,
		key=lambda r: (r.hit_at_10, r.hit_at_5, r.hit_at_1, r.top1_score),
		reverse=True,
	)
	# Worst: prefer misses at @10 and @5, then low score and zero crops.
	bad_sorted = sorted(
		results,
		key=lambda r: (r.hit_at_10, r.hit_at_5, r.hit_at_1, r.top1_score, r.num_crops),
	)

	# Prevent duplicate images in qualitative examples when metadata has
	# multiple rows (for example, multiple garments) for the same image.
	good_sorted = unique_by_image_path(good_sorted)
	bad_sorted = unique_by_image_path(bad_sorted)

	good = good_sorted[:10]
	bad = bad_sorted[:10]

	# Create CSV for good and bad examples (img paths and metrics)
	pd.DataFrame([r.__dict__ for r in good]).to_csv(
		os.path.join(args.output_dir, "good_examples.csv"), index=False
	)
	pd.DataFrame([r.__dict__ for r in bad]).to_csv(
		os.path.join(args.output_dir, "bad_examples.csv"), index=False
	)

	# Create display grids for good and bad examples
	build_grid(good, os.path.join(args.output_dir, "good_examples_grid.png"), "Good Examples")
	build_grid(bad, os.path.join(args.output_dir, "bad_examples_grid.png"), "Bad Examples")

	print(json.dumps(summary, indent=2))
	print(f"Saved outputs to: {args.output_dir}")


def parse_args():
	parser = argparse.ArgumentParser(description="Evaluate full fashion retrieval pipeline.")
	parser.add_argument(
		"--metadata_csv",
		type=str,
		required=True,
		help="Retrieval metadata CSV used to auto-build query rows. i.e. validation_metadata.csv",
	)
	parser.add_argument("--metadata_query_split", type=str, default="query")
	parser.add_argument("--metadata_gallery_split", type=str, default="gallery")
	parser.add_argument("--image_root", type=str, default=os.path.join(PROJECT_ROOT, "ml", "data", "validation", "image"))
	parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_ROOT, "ml", "results", "pipeline_eval"))
	parser.add_argument(
		"--diagnostic_mode",
		type=str,
		default="end_to_end",
		choices=["end_to_end", "localization", "retrieval", "gt_crops", "both"],
		help="Run isolated diagnostics instead of end-to-end evaluation.",
	)

	parser.add_argument("--image_col", type=str, default="image_path")
	parser.add_argument("--gt_col", type=str, default="gt_item_id")
	parser.add_argument(
		"--match_on",
		type=str,
		default="pair_id",
		choices=["pair_id", "item_id"],
		help="Evaluation label target: pair_id (recommended for user->shop retrieval) or item_id.",
	)

	parser.add_argument("--localization_folder", type=str, default="yolo_model")
	parser.add_argument("--embedding_folder", type=str, default="G_finetuning_plus_transfer-learning_ep50")
	parser.add_argument("--embedding_config", type=str, default="G_finetuning_plus_transfer-learning.yaml")
	parser.add_argument(
		"--crop_fusion",
		type=str,
		default="max",
		choices=["max", "avg", "sum"],
		help="How to combine scores from multiple detected crops.",
	)
	parser.add_argument(
		"--diagnostic_features_dir",
		type=str,
		default=os.path.join(PROJECT_ROOT, "ml", "features", "G_finetuning_plus_transfer-learning_projected"),
		help="Projected validation features directory for retrieval diagnostics.",
	)
	
	return parser.parse_args()

# Central Diagnostic function to evaluate localization and retrieval components in isolation (without end-to-end pipeline) using the same query construction from metadata
def diagnose(args, pipeline: FashionRetrievalPipeline, meta_df: pd.DataFrame):
	# Create output dir if not exists
	safe_makedirs(args.output_dir)

	# Create query dataframe
	query_df = meta_df[meta_df["split"] == args.metadata_query_split].reset_index(drop=True)
	if query_df.empty:
		raise ValueError(f"No query rows found in metadata for split '{args.metadata_query_split}'.")

	# If running localization diagnostic, run localization on each query image and evaluate crop counts and missing image rates
	if args.diagnostic_mode in ("localization", "both"):
		loc_summary = diagnose_localization(args, pipeline, query_df)

		# Save localization diagnostic results and print summary
		with open(os.path.join(args.output_dir, "localization_diagnostic.json"), "w", encoding="utf-8") as f:
			json.dump(loc_summary, f, indent=2)
		print(json.dumps({"localization": loc_summary}, indent=2))

	if args.diagnostic_mode in ("retrieval", "both"):
		retrieval_summary = diagnose_retrieval(args, pipeline, query_df)
		with open(os.path.join(args.output_dir, "retrieval_diagnostic.json"), "w", encoding="utf-8") as f:
			json.dump(retrieval_summary, f, indent=2)
		print(json.dumps({"retrieval": retrieval_summary}, indent=2))

	if args.diagnostic_mode in ("gt_crops", "both"):
		gt_summary = diagnose_gt_crops(args, pipeline, query_df)
		with open(os.path.join(args.output_dir, "gt_crops_diagnostic.json"), "w", encoding="utf-8") as f:
			json.dump(gt_summary, f, indent=2)
		print(json.dumps({"gt_crops": gt_summary}, indent=2))

	return None

# Diagnostic function for localization component
def diagnose_localization(args, pipeline: FashionRetrievalPipeline, query_df: pd.DataFrame):
	# Initialize the localization model
	localizer = pipeline.localization_model
	rows_iter = query_df.iterrows()
	if tqdm is not None:
		rows_iter = tqdm(rows_iter, total=len(query_df), desc="Localization diagnostic", unit="img")

	results = []
	missing_images = 0
	zero_crops = 0
	total_crops = 0

	# Loop through all query images and run localization
	for _, row in rows_iter:
		image_path = resolve_image_path(str(row[args.image_col]), args.image_root)
		if not os.path.isfile(image_path):
			missing_images += 1
			results.append({"image_path": image_path, "num_crops": 0, "status": "missing_image"})
			continue

		with open(image_path, "rb") as f:
			cropped_paths = localizer.upload_image(f)

		num_crops = len(cropped_paths)
		total_crops += num_crops
		if num_crops == 0:
			zero_crops += 1

		results.append({
			"image_path": image_path,
			"num_crops": num_crops,
			"status": "ok" if num_crops > 0 else "no_detections",
		})

	n = float(len(results)) if results else 1.0
	summary = {
		"num_queries": int(n),
		"missing_images": int(missing_images),
		"zero_detection_queries": int(zero_crops),
		"zero_detection_rate": float(zero_crops / n),
		"avg_crops_per_query": float(total_crops / n),
		"non_empty_rate": float((n - zero_crops - missing_images) / n),
	}
	pd.DataFrame(results).to_csv(os.path.join(args.output_dir, "localization_diagnostic_rows.csv"), index=False)
	return summary

# Diagnostic function for retrieval component
def diagnose_retrieval(args, pipeline: FashionRetrievalPipeline, query_df: pd.DataFrame):
	# Get feature dir and all npy files
	features_dir = args.diagnostic_features_dir or get_projected_features_dir(args.embedding_folder)
	query_feats_path = os.path.join(features_dir, "query_feats.npy")
	query_item_ids_path = os.path.join(features_dir, "query_item_ids.npy")
	if not os.path.isfile(query_feats_path):
		raise FileNotFoundError(f"Query features not found: {query_feats_path}")
	if not os.path.isfile(query_item_ids_path):
		raise FileNotFoundError(f"Query item ids not found: {query_item_ids_path}")

	query_feats = np.load(query_feats_path, allow_pickle=True).astype(np.float32)
	query_item_ids = np.load(query_item_ids_path, allow_pickle=True).astype(str)
	if len(query_item_ids) == 0:
		raise ValueError(f"No query item ids found in {query_item_ids_path}")

	gallery_item_ids = pipeline.retrieval_model.gallery_item_ids
	if gallery_item_ids is None:
		raise ValueError(
			"Retrieval model has no gallery_item_ids loaded; cannot compute ID-based diagnostics."
		)
	gallery_item_ids = np.asarray(gallery_item_ids).astype(str)
	index_ntotal = int(pipeline.retrieval_model.index.ntotal)
	if index_ntotal != len(gallery_item_ids):
		raise ValueError(
			"Index/gallery metadata mismatch: "
			f"index.ntotal={index_ntotal} but len(gallery_item_ids)={len(gallery_item_ids)}. "
			"Use gallery metadata files from the exact same build that produced the FAISS index."
		)

	# If matching on item id, check for direct item_id overlap between query and gallery
	if args.match_on == "item_id":
		q_set = set(query_item_ids.tolist())
		g_set = set(gallery_item_ids.tolist())
		overlap = len(q_set.intersection(g_set))
		if overlap == 0:
			raise ValueError(
				"No query/gallery item_id overlap. This indicates split or metadata mismatch."
			)
	# If matching on pair_id, check for overlap of pair_ids derived from item_ids between query and gallery
	else:
		q_pairs = set(pair_id_from_item_id(x) for x in query_item_ids.tolist())
		g_pairs = set(pair_id_from_item_id(x) for x in gallery_item_ids.tolist())
		overlap = len(q_pairs.intersection(g_pairs))
		if overlap == 0:
			raise ValueError(
				"No query/gallery pair_id overlap. This indicates split or metadata mismatch."
			)

	# If the query features are correctly computed, get query features by projecting with embedding head (transformer)
	if query_feats.ndim == 3:
		query_feats = pipeline.retrieval_model.project_with_embedding_head(
			dino_feats=query_feats,
			embedding_config_path=pipeline.embedding_config,
			embedding_checkpoint_path=pipeline.embedding_checkpoint,
			project_root=PROJECT_ROOT,
			embedding_device="cuda",
		)

	# Run retrieval for each query feature and compute the hit metrics based on ranked retrieval and ground truth labels
	ranked_lists = pipeline.retrieval_model.search(query_feats, topk=10)
	if len(ranked_lists) != len(query_item_ids):
		raise ValueError(
			f"Query feature count mismatch: {len(ranked_lists)} results vs {len(query_item_ids)} query labels"
		)

	results: List[QueryEvalResult] = []
	for idx, ranked in enumerate(ranked_lists):
		query_item_id = str(query_item_ids[idx])
		gt_label = query_item_id if args.match_on == "item_id" else pair_id_from_item_id(query_item_id)
		top1_item_id = ""
		top1_score = 0.0
		if ranked:
			top1_item_id = str(ranked[0].get("item_id", ""))
			top1_score = float(ranked[0].get("score", 0.0))

		h1 = hit_at_k(ranked, gt_label, 1, args.match_on)
		h5 = hit_at_k(ranked, gt_label, 5, args.match_on)
		h10 = hit_at_k(ranked, gt_label, 10, args.match_on)
		results.append(
			QueryEvalResult(
				image_path=str(query_df.iloc[idx][args.image_col]) if idx < len(query_df) else "",
				gt_item_id=gt_label,
				num_crops=1,
				top1_item_id=top1_item_id,
				top1_score=top1_score,
				hit_at_1=h1,
				hit_at_5=h5,
				hit_at_10=h10,
			)
		)

	n = float(len(results)) if results else 1.0
	return {
		"num_queries": int(n),
		"features_dir": features_dir,
		"index_ntotal": index_ntotal,
		"gallery_item_ids_len": int(len(gallery_item_ids)),
		"label_overlap": int(overlap),
		"match_on": args.match_on,
		"precision@1": sum(precision_at_k(r.hit_at_1, 1) for r in results) / n,
		"precision@5": sum(precision_at_k(r.hit_at_5, 5) for r in results) / n,
		"precision@10": sum(precision_at_k(r.hit_at_10, 10) for r in results) / n,
		"recall@1": sum(r.hit_at_1 for r in results) / n,
		"recall@5": sum(r.hit_at_5 for r in results) / n,
		"recall@10": sum(r.hit_at_10 for r in results) / n,
	}

# Check the ground truth crops for each query - can compare efficacy of localization crops with actual ground truth crops
def diagnose_gt_crops(args, pipeline: FashionRetrievalPipeline, query_df: pd.DataFrame):
	rows_iter = query_df.iterrows()
	if tqdm is not None:
		rows_iter = tqdm(rows_iter, total=len(query_df), desc="GT-crops diagnostic", unit="img")

	results: List[QueryEvalResult] = []
	missing_images = 0
	invalid_boxes = 0

	with tempfile.TemporaryDirectory(prefix="gt_crops_", dir=args.output_dir) as tmp_dir:
		for i, row in rows_iter:

			# Compute the crop based on the ground truth bounding box coordinates
			image_path = resolve_image_path(str(row[args.image_col]), args.image_root)
			if not os.path.isfile(image_path):
				missing_images += 1
				continue

			img = Image.open(image_path).convert("RGB")
			w, h = img.size
			x1 = int(max(0, min(w, row["x1"])))
			y1 = int(max(0, min(h, row["y1"])))
			x2 = int(max(0, min(w, row["x2"])))
			y2 = int(max(0, min(h, row["y2"])))
			if x2 <= x1 or y2 <= y1:
				invalid_boxes += 1
				continue

			crop = img.crop((x1, y1, x2, y2))
			crop_path = os.path.join(tmp_dir, f"gt_{i}.jpg")
			crop.save(crop_path, format="JPEG", quality=95)

			# Get crop results from retrieval model and compute metrics based on ground truth cropping 
			crop_results = pipeline.retrieval_model.search_from_images(
				[crop_path],
				topk=10,
				backbone_name="dinov2_vitb14",
				device="cuda",
				token_mode="cls_patch",
				project_root=PROJECT_ROOT,
				embedding_config_path=pipeline.embedding_config,
				embedding_checkpoint_path=pipeline.embedding_checkpoint,
				embedding_device="cuda",
			)
			ranked = merge_crop_results(crop_results, topk=10, strategy=args.crop_fusion)

			query_item_id = str(row["item_id"])
			gt_label = query_item_id if args.match_on == "item_id" else str(row["pair_id"])
			results.append(
				QueryEvalResult(
					image_path=image_path,
					gt_item_id=gt_label,
					num_crops=1,
					top1_item_id=str(ranked[0].get("item_id", "")) if ranked else "",
					top1_score=float(ranked[0].get("score", 0.0)) if ranked else 0.0,
					hit_at_1=hit_at_k(ranked, gt_label, 1, args.match_on),
					hit_at_5=hit_at_k(ranked, gt_label, 5, args.match_on),
					hit_at_10=hit_at_k(ranked, gt_label, 10, args.match_on),
				)
			)

	n = float(len(results)) if results else 1.0
	return {
		"num_queries": int(n),
		"missing_images": int(missing_images),
		"invalid_boxes": int(invalid_boxes),
		"match_on": args.match_on,
		"crop_fusion": args.crop_fusion,
		"precision@1": sum(precision_at_k(r.hit_at_1, 1) for r in results) / n,
		"precision@5": sum(precision_at_k(r.hit_at_5, 5) for r in results) / n,
		"precision@10": sum(precision_at_k(r.hit_at_10, 10) for r in results) / n,
		"recall@1": sum(r.hit_at_1 for r in results) / n,
		"recall@5": sum(r.hit_at_5 for r in results) / n,
		"recall@10": sum(r.hit_at_10 for r in results) / n,
	}


if __name__ == "__main__":
	evaluate(parse_args())
