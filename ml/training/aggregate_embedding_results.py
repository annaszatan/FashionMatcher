"""
Aggregate embedding-input experiment results.
Outputs:
- results/aggregated_embedding_results/embedding_architecture_comparison.csv/.md
- results/aggregated_embedding_results/embedding_loss_comparison.csv/.md
- results/aggregated_embedding_results/plots/embedding/<loss>/{embedding|scratch}_Recall_*.png (one bar chart per loss folder)
"""
import os
import argparse
import json

import pandas as pd


def load_rows(results_dir: str):
    rows = []
    if not os.path.isdir(results_dir):
        return rows
    for exp_name in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, exp_name, "metrics.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r") as f:
                m = json.load(f)
            if not isinstance(m, dict):
                continue
            m["experiment_name"] = m.get("experiment_name", exp_name)
            rows.append(m)
        except Exception:
            continue
    return rows


LOSS_TYPES = ("supcon", "triplet", "infonce")


def _normalize_loss_type(lt) -> str:
    s = str(lt).strip().lower()
    if s in ("nt_xent", "ntxent"):
        return "infonce"
    return s


def _loss_type_title(loss_key: str) -> str:
    """Plot subtitle segment, e.g. infonce -> Infonce."""
    return str(loss_key).strip().lower().title()


def _is_scratch_experiment(name: str) -> bool:
    """E/F embedding scratch runs (train_embedding_model configs)."""
    s = str(name)
    return s.startswith("E_embedding_") or s.startswith("F_embedding_")


def write_table(df: pd.DataFrame, csv_path: str, md_path: str, title: str):
    df.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write(df.to_string(index=False))
        f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/experiments")
    parser.add_argument("--output_dir", type=str, default="results/aggregated_embedding_results")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument(
        "--scratch",
        action="store_true",
        help="Use only E_embedding_* and F_embedding_* experiments (tables + plots). "
        "Writes *_scratch.csv/.md and plots under plots/embedding_scratch/.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    rows = load_rows(results_dir)
    if not rows:
        print(f"No metrics found under {results_dir}")
        return

    df = pd.DataFrame(rows)
    # Keep embedding-based experiments only
    bb = df.get("backbone", "").astype(str)
    mask_embed = bb.str.contains("dinov2_frozen", na=False) | bb.str.contains("dinov2_lora", na=False)
    df_embed = df[mask_embed].copy() if "backbone" in df.columns else pd.DataFrame()
    if df_embed.empty:
        print("No embedding-input experiment rows found.")
        return

    if args.scratch:
        names = df_embed["experiment_name"].astype(str)
        scratch_mask = names.map(_is_scratch_experiment)
        df_embed = df_embed[scratch_mask].copy()
        if df_embed.empty:
            print("No E_embedding_* / F_embedding_* experiments found (--scratch).")
            return

    out_suffix = "_scratch" if args.scratch else ""
    plot_subdir = "embedding_scratch" if args.scratch else "embedding"

    # Architecture comparison (transformer vs cnn, default supcon or all if present)
    arch_df = df_embed[df_embed["training_strategy"].isin(["transformer", "cnn"])].copy()
    arch_cols = [
        "experiment_name", "backbone", "training_strategy", "loss_type", "trainable_params",
        "Recall@1", "Recall@5", "Recall@10", "mAP@10", "mAP@50", "best_epoch",
    ]
    for c in arch_cols:
        if c not in arch_df.columns:
            arch_df[c] = ""
    arch_df = arch_df[arch_cols]
    arch_df["_loss_norm"] = arch_df["loss_type"].map(_normalize_loss_type)
    write_table(
        arch_df.drop(columns=["_loss_norm"], errors="ignore"),
        os.path.join(output_dir, f"embedding_architecture_comparison{out_suffix}.csv"),
        os.path.join(output_dir, f"embedding_architecture_comparison{out_suffix}.md"),
        "Embedding Architecture Comparison" + (" (scratch E/F)" if args.scratch else ""),
    )

    # Loss comparison (supcon/triplet/infonce on transformer by default)
    loss_df = df_embed[df_embed["loss_type"].astype(str).str.lower().isin(["supcon", "triplet", "infonce"])].copy()
    loss_cols = [
        "experiment_name", "backbone", "training_strategy", "loss_type",
        "Recall@1", "Recall@5", "Recall@10", "mAP@10", "mAP@50", "best_epoch",
    ]
    for c in loss_cols:
        if c not in loss_df.columns:
            loss_df[c] = ""
    loss_df = loss_df[loss_cols]
    loss_df["_loss_norm"] = loss_df["loss_type"].map(_normalize_loss_type)
    write_table(
        loss_df.drop(columns=["_loss_norm"], errors="ignore"),
        os.path.join(output_dir, f"embedding_loss_comparison{out_suffix}.csv"),
        os.path.join(output_dir, f"embedding_loss_comparison{out_suffix}.md"),
        "Embedding Loss Comparison" + (" (scratch E/F)" if args.scratch else ""),
    )

    if args.plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Match aggregate_results.py (A~D transfer / loss plots)
            plt.rcParams.update({
                "figure.facecolor": "#FAFAFA",
                "axes.facecolor": "#FAFAFA",
                "axes.edgecolor": "#CCCCCC",
                "axes.grid": True,
                "grid.alpha": 0.3,
                "grid.linestyle": "--",
                "font.family": "sans-serif",
                "font.size": 11,
                "axes.titlesize": 14,
                "axes.titleweight": "bold",
                "axes.labelsize": 12,
            })

            PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

            plot_base = os.path.join(output_dir, "plots", plot_subdir)
            os.makedirs(plot_base, exist_ok=True)
            recall_metrics = ["Recall@1", "Recall@5", "Recall@10"]

            def plot_grouped_bars(df, metrics, title_prefix, palette, filename_prefix, plots_dir):
                """Same bar design as scripts/aggregate_results.py plot_grouped_bars."""
                if df is None or df.empty:
                    return
                df = df.drop(columns=["_loss_norm"], errors="ignore")
                for col in metrics:
                    if col not in df.columns:
                        continue
                    vals = pd.to_numeric(df[col], errors="coerce")
                    if vals.isna().all():
                        continue

                    names = df["experiment_name"].tolist()
                    colors = [palette[i % len(palette)] for i in range(len(names))]

                    fig_w = max(8.0, len(names) * 1.85)
                    fig_h = 6.8 if len(names) > 4 else 6.2
                    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                    bars = ax.bar(
                        range(len(names)),
                        vals,
                        color=colors,
                        width=0.6,
                        edgecolor="white",
                        linewidth=1.2,
                        zorder=3,
                    )

                    for bar, v in zip(bars, vals):
                        if pd.notna(v):
                            ax.annotate(
                                f"{v:.3f}",
                                xy=(bar.get_x() + bar.get_width() / 2, v),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha="center",
                                va="bottom",
                                fontsize=10,
                                fontweight="bold",
                                color="#333333",
                            )

                    ax.set_xticks(range(len(names)))
                    ax.set_xticklabels(names, rotation=0, ha="center", fontsize=10)
                    ax.set_ylabel(col)
                    ax.set_title(f"{title_prefix} ({col})")
                    vmax = float(vals.max())
                    if pd.isna(vmax) or vmax <= 0:
                        y_top = 1.0
                    else:
                        headroom = 0.14
                        y_top = vmax * (1.0 + headroom)
                        if vmax <= 1.0:
                            y_top = min(y_top, 1.12)
                    ax.set_ylim(0, y_top)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)

                    fig.tight_layout(pad=1.2)
                    fname = f"{filename_prefix}_{col.replace('@', '_')}.png"
                    fig.savefig(
                        os.path.join(plots_dir, fname),
                        dpi=150,
                        bbox_inches="tight",
                        pad_inches=0.35,
                    )
                    plt.close(fig)

            plot_name_prefix = "scratch" if args.scratch else "embedding"
            for loss_type in LOSS_TYPES:
                loss_plots_dir = os.path.join(plot_base, loss_type)
                os.makedirs(loss_plots_dir, exist_ok=True)

                sub = df_embed.copy()
                sub["_loss_norm"] = sub["loss_type"].map(_normalize_loss_type)
                filtered = sub[sub["_loss_norm"] == loss_type].copy()
                title = f"Transfer Learning ({_loss_type_title(loss_type)})"
                plot_grouped_bars(
                    filtered, recall_metrics, title, PALETTE, plot_name_prefix, loss_plots_dir
                )

            print(f"Plots written under {plot_base}/<supcon|triplet|infonce>/ ({plot_name_prefix}_Recall_*.png)")
        except Exception as e:
            print(f"Plot skipped: {e}")

    print(f"Wrote {os.path.join(output_dir, f'embedding_architecture_comparison{out_suffix}.csv')}")
    print(f"Wrote {os.path.join(output_dir, f'embedding_loss_comparison{out_suffix}.csv')}")


if __name__ == "__main__":
    main()
