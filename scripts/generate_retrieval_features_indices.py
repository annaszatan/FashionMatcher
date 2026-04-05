"""
This file will help you generate the features and FAISS index required for retrieval with a trained checkpoint
Example usage:
```
python scripts/generate_retrieval_features_indices.py \
    --config ml/configs/C_lora_peft.yaml \
    --checkpoint_dir ml/checkpoints/C_lora_peft \
    --output_tokens cls_patch \
    --projection_config ml/configs/G_finetuning_transformer_projection.yaml \
    --projection_checkpoint ml/checkpoints/G_finetuning_transformer_ep50 \
    --projection_batch_size 128
"""

import argparse
import os
import subprocess
import sys
import tempfile

import yaml


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Make sure path is fixed
def resolve(path: str) -> str:
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))

# Run a subprocess command with logging and error handling
def run(cmd):
    print("[CMD] " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

# Extract a checkpoint name from the checkpoint directory for naming outputs
def checkpoint_name_from_dir(checkpoint_dir: str) -> str:
    name = os.path.basename(os.path.normpath(checkpoint_dir))
    return name if name else "checkpoint"


def resolve_checkpoint(path_or_dir: str) -> str:
    """Accept either a checkpoint file path or a directory containing best.pt."""
    p = resolve(path_or_dir)
    if os.path.isdir(p):
        p = os.path.join(p, "best.pt")
    return p


def write_projection_config(base_config_path: str, input_features_dir: str) -> str:
    """Create a temporary projection config with paths.input_features_dir overridden."""
    with open(base_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Projection config must parse to a dict: {base_config_path}")

    paths = cfg.get("paths", {})
    if not isinstance(paths, dict):
        paths = {}
    paths["input_features_dir"] = input_features_dir
    cfg["paths"] = paths

    fd, tmp_path = tempfile.mkstemp(prefix="proj_cfg_", suffix=".yaml")
    os.close(fd)
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return tmp_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate LoRA retrieval tokens and FAISS index from a config + checkpoint directory."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Training config yaml used by extract_lora_retrieval_tokens.py",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Checkpoint directory containing best.pt (for example ml/checkpoints/C_lora_peft)",
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default=None,
        help="Output directory for token features. Default: ml/features/<checkpoint_name>_<output_tokens>",
    )
    parser.add_argument(
        "--output_index",
        type=str,
        default=None,
        help="Output FAISS index path. Default: ml/indices/<checkpoint_name>_<output_tokens>/faiss_gallery_index.ip",
    )
    parser.add_argument(
        "--output_tokens",
        type=str,
        default="cls_patch",
        choices=["cls_patch", "projection"],
        help="Token type to extract for retrieval features",
    )
    parser.add_argument(
        "--projection_config",
        type=str,
        default=None,
        help="Embedding projection config YAML (required when --output_tokens=cls_patch)",
    )
    parser.add_argument(
        "--projection_checkpoint",
        type=str,
        default=None,
        help="Embedding projection checkpoint path or directory containing best.pt (required when --output_tokens=cls_patch)",
    )
    parser.add_argument(
        "--projected_features_dir",
        type=str,
        default=None,
        help="Where projected features are saved for cls_patch mode. Default: <features_dir>_projected",
    )
    parser.add_argument(
        "--projection_batch_size",
        type=int,
        default=256,
        help="Batch size for projection stage in cls_patch mode",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Check for required files and directories
    config_path = resolve(args.config)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    checkpoint_dir = resolve(args.checkpoint_dir)
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Get output dir names and create if non-existent
    ckpt_name = checkpoint_name_from_dir(checkpoint_dir)
    default_features_dir = os.path.join("ml", "features", f"{ckpt_name}")
    features_dir = resolve(args.features_dir) if args.features_dir else resolve(default_features_dir)
    os.makedirs(features_dir, exist_ok=True)

    default_index_subdir = ckpt_name if args.output_tokens == "projection" else f"{ckpt_name}_projected"
    default_index_path = os.path.join("ml", "indices", default_index_subdir, "faiss_gallery_index.ip")
    output_index = resolve(args.output_index) if args.output_index else resolve(default_index_path)
    os.makedirs(os.path.dirname(output_index), exist_ok=True)

    # Get paths to required scripts in preprocessing folder
    extract_script = os.path.join(PROJECT_ROOT, "ml", "preprocessing", "extract_lora_retrieval_tokens.py")
    index_script = os.path.join(PROJECT_ROOT, "ml", "preprocessing", "build_faiss_index.py")
    projected_index_script = os.path.join(PROJECT_ROOT, "ml", "preprocessing", "build_projected_faiss_index.py")

    if not os.path.isfile(extract_script):
        raise FileNotFoundError(f"Extractor script not found: {extract_script}")
    if not os.path.isfile(index_script):
        raise FileNotFoundError(f"Index builder script not found: {index_script}")
    if not os.path.isfile(projected_index_script):
        raise FileNotFoundError(f"Projected index builder script not found: {projected_index_script}")

    # Extract features using extraction script (lora tokens)
    run(
        [
            sys.executable,
            extract_script,
            "--config",
            config_path,
            "--checkpoint",
            checkpoint_path,
            "--features_dir",
            features_dir,
            "--output_tokens",
            args.output_tokens,
        ]
    )

    gallery_feats = os.path.join(features_dir, "gallery_feats.npy")
    if not os.path.isfile(gallery_feats):
        raise FileNotFoundError(f"Expected gallery features not found: {gallery_feats}")

    # If output tokens are projections (i.e. mlp), we can directly build the FAISS index
    if args.output_tokens == "projection":
        run(
            [
                sys.executable,
                index_script,
                "--emb_path",
                gallery_feats,
                "--output",
                output_index,
            ]
        )
    # If the output tokens are cls_patch, we need to project the embeddings with the projection head (i.e. Transformer) first before building the FAISS index
    else:
        if not args.projection_config:
            raise ValueError("--projection_config is required when --output_tokens=cls_patch")
        if not args.projection_checkpoint:
            raise ValueError("--projection_checkpoint is required when --output_tokens=cls_patch")

        # Make sure that projection head config and checkpoint exist before starting projection + indexing
        projection_config_path = resolve(args.projection_config)
        projection_checkpoint_path = resolve_checkpoint(args.projection_checkpoint)
        if not os.path.isfile(projection_config_path):
            raise FileNotFoundError(f"Projection config not found: {projection_config_path}")
        if not os.path.isfile(projection_checkpoint_path):
            raise FileNotFoundError(f"Projection checkpoint not found: {projection_checkpoint_path}")

        # Create projected features directory
        projected_features_dir = (
            resolve(args.projected_features_dir)
            if args.projected_features_dir
            else f"{features_dir}_projected"
        )
        os.makedirs(projected_features_dir, exist_ok=True)
        
        # Create a temporary projection config with the correct input feature paths and run the faiss projection index
        tmp_cfg = write_projection_config(
            base_config_path=projection_config_path,
            input_features_dir=features_dir,
        )
        try:
            run(
                [
                    sys.executable,
                    projected_index_script,
                    "--config",
                    tmp_cfg,
                    "--checkpoint",
                    projection_checkpoint_path,
                    "--output_features_dir",
                    projected_features_dir,
                    "--output_index",
                    output_index,
                    "--batch_size",
                    str(args.projection_batch_size),
                ]
            )
        finally:
            # Remove temporary config file
            if os.path.isfile(tmp_cfg):
                os.remove(tmp_cfg)

    print("[DONE] LoRA token features + FAISS index generated successfully.")
    print(f"[OUT] features_dir={features_dir}")
    if args.output_tokens == "cls_patch":
        print(f"[OUT] projected_features_dir={projected_features_dir}")
    print(f"[OUT] index_path={output_index}")


if __name__ == "__main__":
    main()
