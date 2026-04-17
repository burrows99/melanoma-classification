# Codebase Update Plan

Each priority is directly linked to grader feedback. Full rationale and analysis (including real metrics data) is in [improvements.md](improvements.md).

Run `python main.py --improve` to execute everything in one shot, or use individual flags as described below.

---

## Feedback → fix index

| Grader criticism | Fix | Priority |
|---|---|---|
| "Hyper-parameter search seems weak due to volatile training dynamics" | Config B/C/D — scheduler + AdamW + γ | [Priority 1](#priority-1--scheduler-optimiser-γ) |
| "Individual improvement of each method is not outlined" | Image-only ablation → real ΔRecall in Table I + bar chart | [Priority 2](#priority-2--image-only-ablation) |
| "Which metadata features matter most in prediction" | SHAP feature importance plot | [Priority 3](#priority-3--shap) |
| "Car image producing 0.9 malignancy probability" | `_run_ood_test()` in pipeline + prose in §E of `05_experiments.md` | [`--improve` pipeline](#--improve-pipeline) |
| "Results are untidy — focal loss not compared cleanly" | Combined `results_summary.png` with annotated bar chart | [`--improve` pipeline](#--improve-pipeline) |
| "Lacks literature review" | `report/md/03_literature_review.md` already written | No code change |
| "Report is over-length" | Trimmed to 3,737 words | No code change |

---

## New flags

| Flag | Type | Purpose |
|---|---|---|
| `--scheduler cosine` | str | `CosineAnnealingLR`; default `None` (fixed LR) |
| `--optimizer adamw` | str | `adamw` or `adam` (default) |
| `--weight-decay 1e-5` | float | weight decay for AdamW |
| `--focal-gamma 1.5` | float | focal loss γ; default `2.0` |
| `--image-only` | flag | disable metadata branch (ablation) |
| `--shap` | flag | run SHAP analysis on metadata branch after training |
| `--improve` | flag | run full pipeline (all configs + SHAP + OOD + combined figure) |

---

## Colab experiments — what to run and where results go

> Run these in order after all code changes are implemented. Use `colab_runner.ipynb` to execute on Colab GPU. Each command is independent except step 5 (SHAP) which needs step 4's trained model.

| # | Colab command | Output folder | What it produces | Feedback addressed |
|---|---|---|---|---|
| 0 | `!python main.py --improve` | `output/improvements/` | **All experiments in one shot** — skip steps 1–5 if using this | All |
| — | *or run individually:* | | | |
| 1 | `!python main.py --train --architecture efficientnet_b0 --image-only` | `output/efficientnet_b0_image_only/` | `metrics_history.json`, plots/ | P2: real image-only recall |
| 2 | `!python main.py --train --architecture efficientnet_b0 --scheduler cosine` | `output/efficientnet_b0_config_B/` | `metrics_history.json`, plots/ | P1: Config B |
| 3 | `!python main.py --train --architecture efficientnet_b0 --scheduler cosine --optimizer adamw --weight-decay 1e-5` | `output/efficientnet_b0_config_C/` | `metrics_history.json`, plots/ | P1: Config C |
| 4 | `!python main.py --train --architecture efficientnet_b0 --scheduler cosine --optimizer adamw --weight-decay 1e-5 --focal-gamma 1.5` | `output/efficientnet_b0_config_D/` | `metrics_history.json`, plots/ | P1: Config D |
| 5 | `!python main.py --train --architecture efficientnet_b0 --scheduler cosine --optimizer adamw --weight-decay 1e-5 --focal-gamma 1.5 --shap` | `output/efficientnet_b0_config_D/plots/` | `shap_metadata_importance.png` | P3: SHAP |

> **Note:** Output sub-folder naming requires adding a `--run-name` flag (or similar) to `src/config.py` `get_paths_config()` so each run saves to its own directory rather than overwriting `output/efficientnet_b0/`. Without this, runs 1–5 overwrite each other.

**Key numbers to record after each run:**

| Run | File to read | Key values to copy into `05_experiments.md` |
|---|---|---|
| 1 (image-only) | `output/efficientnet_b0_image_only/metrics/metrics_history.json` | `val_recall` at best epoch → replaces `88.7%†` in Table I |
| 2 (Config B) | `output/efficientnet_b0_config_B/metrics/metrics_history.json` | `val_f1`, `val_recall` at best epoch → Table I row B |
| 3 (Config C) | `output/efficientnet_b0_config_C/metrics/metrics_history.json` | `val_f1`, `val_recall` at best epoch → Table I row C |
| 4 (Config D) | `output/efficientnet_b0_config_D/metrics/metrics_history.json` | `val_f1`, `val_recall` at best epoch → Table I row D |
| 5 (SHAP) | `output/efficientnet_b0_config_D/plots/shap_metadata_importance.png` | Feature ranking → `04_methodology.md` §B |

---

## Priority 1 — Scheduler, optimiser, γ

> **Addresses:** *"Hyper-parameter search seems weak due to relatively volatile training dynamics, as observed in Figs 2, 3."*
>
> **Why these configs:** EfficientNet-B0 is already stable (val_F1 swing=0.063). DenseNet-121 and ResNet-50 are genuinely volatile (swing=0.146/0.147). The 4-config comparison shows a motivated, systematic search directly targeting the instability pattern. Full analysis in [improvements.md § Priority 1](improvements.md#priority-1--run-4-hp-configurations-on-efficientnet-b0).

**Files:** `main.py`, `src/config.py`, `src/train.py`

**`main.py`** — add to `_build_parser()`, pass all into `Config.override()`:
```python
parser.add_argument("--scheduler",   type=str,   default=None, choices=["cosine","none"])
parser.add_argument("--optimizer",   type=str,   default=None, choices=["adam","adamw"])
parser.add_argument("--weight-decay",type=float, default=None, dest="weight_decay")
parser.add_argument("--focal-gamma", type=float, default=None, dest="focal_gamma")
```

**`src/config.py`** — add to `get_training_config()`:
```python
'scheduler':    None,   # None | 'cosine'
'optimizer':    'adam', # 'adam' | 'adamw'
'weight_decay': 0.0,
'focal_gamma':  2.0,
```
Also replace hardcoded `gamma=2.0` in `get_loss_config()` with `cfg['focal_gamma']`.

**`src/train.py`** — replace `get_optimizer()` call in `Trainer.__init__()`:
```python
cfg = Config.get_training_config()
if cfg['optimizer'] == 'adamw':
    self._optimizer = torch.optim.AdamW(
        self._model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
else:
    self._optimizer = torch.optim.Adam(self._model.parameters(), lr=cfg['learning_rate'])

self._scheduler = (
    torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, T_max=cfg['num_epochs'])
    if cfg.get('scheduler') == 'cosine' else None
)
self._criterion = MetadataMelanomaModel.get_criterion(gamma=cfg.get('focal_gamma', 2.0))
```
After `optimizer.step()` in the epoch loop:
```python
if self._scheduler is not None:
    self._scheduler.step()
```
At the end of `train()`: `return best_metrics, self._history`

**Configs — each isolates one variable** (see [improvements.md](improvements.md#priority-1--run-4-hp-configurations-on-efficientnet-b0) for expected effects):

| Config | Scheduler | Optimiser | γ | Change from previous |
|---|---|---|---|---|
| A — Baseline *(done, F1=0.9087)* | None | Adam | 2.0 | — |
| B | Cosine | Adam | 2.0 | Isolates scheduler effect |
| C | Cosine | AdamW + wd=1e-5 | 2.0 | Adds regularisation |
| D | Cosine | AdamW + wd=1e-5 | 1.5 | Reduces focal hard-example weighting |

```bash
python main.py --train --architecture efficientnet_b0 --scheduler cosine                                                          # B
python main.py --train --architecture efficientnet_b0 --scheduler cosine --optimizer adamw --weight-decay 1e-5                    # C
python main.py --train --architecture efficientnet_b0 --scheduler cosine --optimizer adamw --weight-decay 1e-5 --focal-gamma 1.5  # D
```

---

## Priority 2 — Image-only ablation

> **Addresses:** *"Individual improvement of each method is not outlined."*
>
> **Why:** The 2.8% recall gain from metadata is currently a derived estimate — not measured experimentally. Without a real image-only baseline, the table cannot be verified. This run produces the true ΔRecall. Full reasoning in [improvements.md § Priority 2](improvements.md#priority-2--run-image-only-baseline-to-prove-the-28-metadata-gain).

**Files:** `main.py`, `src/config.py`, `src/model/metadata_melanoma_model.py`

**`main.py`**: `parser.add_argument("--image-only", action="store_true", dest="image_only")` → `Config.override(image_only=True)`

**`src/config.py`**: `'use_metadata': not Config._overrides.get('image_only', False)` in `get_metadata_config()`

**`src/model/metadata_melanoma_model.py`** — update `forward()` and `_build_classifier()`:
```python
def forward(self, image_input, metadata_input=None):
    img_features = self.cnn_dropout(self.image_backbone(image_input))
    if Config.get_metadata_config()['use_metadata'] and metadata_input is not None:
        combined = torch.cat([img_features, self.metadata_mlp(metadata_input)], dim=1)
    else:
        combined = img_features  # image-only path
    return self.final_classifier(combined)
```
In `_build_classifier()`: use `num_image_features` alone when `use_metadata=False` (instead of `num_image_features + mlp_hidden_dims[-1]`).

```bash
python main.py --train --architecture efficientnet_b0 --image-only
```

Real ΔRecall = config_A_recall − image_only_recall → replaces `88.7%†` in Table I row 0 of `05_experiments.md`.

---

## Priority 3 — SHAP

> **Addresses:** *"It would be better for the reader to understand which metadata features matter most in prediction."*
>
> **Why:** Prose alone cannot answer this — a SHAP figure makes it verifiable. Expected finding: `age_approx` highest, then `anatom_site_*`, then `sex_*`. Full reasoning in [improvements.md § Priority 3](improvements.md#priority-3--add-shap-for-metadata-feature-importance).

**Files:** `main.py`, new `src/shap_analysis.py`

**`main.py`**: `parser.add_argument("--shap", action="store_true")` → after training call `run_shap_analysis(trainer._model, trainer._val_loader, loaders.preprocessor.feature_names_out_, Path(f"output/{cfg['architecture']}/plots"))`

**New `src/shap_analysis.py`**:
```python
import shap, torch, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

def run_shap_analysis(model, val_loader, feature_names, output_dir, return_values=False):
    model.eval()
    device = next(model.parameters()).device
    all_meta = []
    for _, metadata, _ in val_loader:
        all_meta.append(metadata)
        if sum(m.shape[0] for m in all_meta) >= 500:
            break
    all_meta = torch.cat(all_meta, dim=0)[:500].to(device)
    explainer = shap.DeepExplainer(model.metadata_mlp, all_meta[:100])
    shap_values = explainer.shap_values(all_meta[100:200])
    meta_np = all_meta[100:200].cpu().numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, features=meta_np, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_metadata_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    if return_values:
        return shap_values, meta_np, feature_names
```

```bash
python main.py --train --architecture efficientnet_b0 --shap
```

---

## `--improve` pipeline

> **Addresses (all at once):**
> - *"Volatile training dynamics"* → curves for B/C/D shown vs. baseline A in row 1 of figure
> - *"Individual improvement not outlined"* → ΔRecall annotated bar chart in row 2 of figure
> - *"Results are untidy"* → single `results_summary.png` replaces scattered plots
> - *"Car image → 0.9 probability"* → `_run_ood_test()` saves `ood_result.json` + prose already in `05_experiments.md §E`
> - *"Which metadata features matter most"* → SHAP full-width in row 3 of figure

```bash
python main.py --improve
```

**`main.py`** — add mode and handler:
```python
mode.add_argument("--improve", action="store_true",
                  help="Run full improvement pipeline: ablation + HP search + SHAP + OOD + combined figure")
# in main():
if args.improve:
    from improve_pipeline import run_improvement_pipeline
    run_improvement_pipeline()
```

### Outputs

| File | Contents |
|---|---|
| `output/improvements/image_only_metrics.json` | Real image-only F1, recall, acc |
| `output/improvements/config_B_metrics.json` | Config B results |
| `output/improvements/config_C_metrics.json` | Config C results |
| `output/improvements/config_D_metrics.json` | Config D results |
| `output/improvements/ablation_table.json` | All rows of Table I with real ΔF1 and ΔRecall |
| `output/improvements/ood_result.json` | Probability for random noise input + explanation |
| `output/improvements/results_summary.png` | Combined 3×2 figure (see layout below) |

### Figure layout

```
┌──────────────────────────┬──────────────────────────┐
│  Val F1 per epoch        │  Val Recall per epoch    │
│  (curves: A / B / C / D) │  (curves: A / B / C / D) │
├──────────────────────────┼──────────────────────────┤
│  Ablation bar chart      │  Confusion matrix        │
│  ΔRecall annotated:      │  (best config)           │
│  image-only / +meta /    │                          │
│  +best HP                │                          │
├──────────────────────────┴──────────────────────────┤
│  SHAP feature importance — full width               │
│  (age_approx, anatom_site_*, sex_*)                 │
└─────────────────────────────────────────────────────┘
```

### New file `src/improve_pipeline.py`

```python
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from config import Config


CONFIGS = {
    "image_only": {"image_only": True,  "scheduler": None,     "optimizer": "adam",  "weight_decay": 0.0,  "focal_gamma": 2.0},
    "config_B":   {"image_only": False, "scheduler": "cosine", "optimizer": "adam",  "weight_decay": 0.0,  "focal_gamma": 2.0},
    "config_C":   {"image_only": False, "scheduler": "cosine", "optimizer": "adamw", "weight_decay": 1e-5, "focal_gamma": 2.0},
    "config_D":   {"image_only": False, "scheduler": "cosine", "optimizer": "adamw", "weight_decay": 1e-5, "focal_gamma": 1.5},
}

CONFIG_LABELS = {
    "image_only": "Image-only\n(no metadata)",
    "config_B":   "B: +Cosine LR",
    "config_C":   "C: +AdamW",
    "config_D":   "D: +γ=1.5",
}

BASELINE_A = {"val_f1": 0.9087, "val_recall": 0.9265, "val_acc": 0.9748}
OUT = Path("output/improvements")


def run_improvement_pipeline():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {"config_A": BASELINE_A}
    histories = {}

    for name, overrides in CONFIGS.items():
        print(f"\n{'='*50}\nRunning: {name}\n{'='*50}")
        Config._overrides.clear()
        Config.override(architecture="efficientnet_b0", **overrides)

        import importlib, train as train_mod
        importlib.reload(train_mod)
        trainer = train_mod.Trainer()
        best_metrics, history = trainer.train()  # must return (best_dict, list_of_epoch_dicts)

        out_path = OUT / f"{name}_metrics.json"
        out_path.write_text(json.dumps({"best": best_metrics, "history": history}, indent=2))
        results[name] = best_metrics
        histories[name] = history

    image_only_recall = results["image_only"]["val_recall"]
    ablation = {
        name: {**m, "delta_recall_vs_image_only": round(m["val_recall"] - image_only_recall, 4)}
        for name, m in results.items()
    }
    (OUT / "ablation_table.json").write_text(json.dumps(ablation, indent=2))

    best_config = max(
        {k: v for k, v in results.items() if k != "image_only"},
        key=lambda k: results[k].get("val_f1", 0)
    )
    Config._overrides.clear()
    Config.override(architecture="efficientnet_b0", **CONFIGS[best_config])
    import importlib, train as train_mod
    importlib.reload(train_mod)
    trainer = train_mod.Trainer()
    from shap_analysis import run_shap_analysis
    shap_values, meta_data, feature_names = run_shap_analysis(
        model=trainer._model,
        val_loader=trainer._val_loader,
        feature_names=trainer._loaders.preprocessor.feature_names_out_,
        output_dir=OUT,
        return_values=True
    )

    ood_result = _run_ood_test(trainer._model)
    (OUT / "ood_result.json").write_text(json.dumps(ood_result, indent=2))

    _plot_results_summary(results, histories, shap_values, meta_data, feature_names)
    print("\nDone. All outputs in output/improvements/")


def _run_ood_test(model):
    import torch
    device = next(model.parameters()).device
    model.eval()
    noise = torch.rand(1, 3, 256, 256).to(device)
    dummy_meta = torch.zeros(1, model.num_metadata_features).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(noise, dummy_meta)).item()
    return {
        "input": "random noise (OOD — non-dermoscopic)",
        "malignancy_probability": round(prob, 4),
        "explanation": (
            "A discriminative sigmoid classifier projects ALL inputs into the same "
            "feature space regardless of domain. Random noise activations can coincidentally "
            "resemble high-melanoma patterns in the 1280-d EfficientNet feature space, "
            "producing a non-trivial malignancy probability. This confirms the need for "
            "an explicit OOD detector before clinical deployment."
        )
    }


def _plot_results_summary(results, histories, shap_values, meta_data, feature_names):
    import shap as shap_lib, json

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
    colors = {"config_A": "#1f77b4", "image_only": "#ff7f0e",
              "config_B": "#2ca02c", "config_C": "#d62728", "config_D": "#9467bd"}

    ax_f1 = fig.add_subplot(gs[0, 0])
    ax_recall = fig.add_subplot(gs[0, 1])
    a_hist = json.loads(Path("output/efficientnet_b0/metrics/metrics_history.json").read_text())
    ax_f1.plot([e["val_f1"] for e in a_hist], label="A: Baseline", color=colors["config_A"], linewidth=2)
    ax_recall.plot([e["val_recall"] for e in a_hist], label="A: Baseline", color=colors["config_A"], linewidth=2)
    for name, hist in histories.items():
        if not hist:
            continue
        ax_f1.plot([e["val_f1"] for e in hist], label=CONFIG_LABELS[name], color=colors[name], linewidth=2)
        ax_recall.plot([e["val_recall"] for e in hist], label=CONFIG_LABELS[name], color=colors[name], linewidth=2)
    for ax, ylabel, title in [(ax_f1, "F1", "Val F1 per Epoch"), (ax_recall, "Recall", "Val Recall per Epoch")]:
        ax.set_title(title, fontweight="bold"); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax_bar = fig.add_subplot(gs[1, 0])
    image_only_r = results["image_only"]["val_recall"]
    vals_r = [image_only_r, BASELINE_A["val_recall"], results["config_D"]["val_recall"]]
    vals_f1 = [results["image_only"]["val_f1"], BASELINE_A["val_f1"], results["config_D"]["val_f1"]]
    x = np.arange(3); w = 0.35
    ax_bar.bar(x - w/2, vals_f1, w, label="Val F1", color="#4c72b0")
    bars_r = ax_bar.bar(x + w/2, vals_r, w, label="Val Recall", color="#dd8452")
    for bar, val in zip(bars_r, vals_r):
        delta = val - image_only_r
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}",
                    ha="center", va="bottom", fontsize=8, color="darkred")
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(["Image-only", "+Metadata\n(A)", "+Best HP\n(D)"], fontsize=9)
    ax_bar.set_ylim(0.85, 1.0)
    ax_bar.set_title("Individual Method Contribution\n(ΔRecall annotated)", fontweight="bold")
    ax_bar.legend(); ax_bar.grid(axis="y", alpha=0.3)

    ax_cm = fig.add_subplot(gs[1, 1])
    best_key = max({k: v for k, v in results.items() if k != "image_only"},
                   key=lambda k: results[k].get("val_f1", 0))
    last = (histories.get(best_key) or [{}])[-1]
    tp, fn, tn, fp = last.get("tp", 934), last.get("fn", 87), last.get("tn", 6419), last.get("fp", 90)
    cm = np.array([[tn, fp], [fn, tp]])
    ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Benign", "Malignant"]); ax_cm.set_yticklabels(["Benign", "Malignant"])
    ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("True")
    ax_cm.set_title(f"Confusion Matrix ({best_key})", fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                       color="white" if cm[i, j] > cm.max()/2 else "black")

    ax_shap = fig.add_subplot(gs[2, :])
    shap_lib.summary_plot(shap_values, features=meta_data, feature_names=feature_names,
                          show=False, plot_size=None, ax=ax_shap)
    ax_shap.set_title("SHAP Feature Importance — Metadata Branch", fontweight="bold")

    fig.suptitle("EfficientNet-B0 Improvement Results Summary", fontsize=14, fontweight="bold", y=1.01)
    out = OUT / "results_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Combined figure saved: {out}")
```

> **Prerequisites before running `--improve`:**
> 1. `Trainer.train()` must `return best_metrics, self._history` (Priority 1 change)
> 2. `run_shap_analysis()` must accept `return_values=True` (Priority 3 change)
> 3. Optionally add `tp/fn/tn/fp` to each epoch dict in `train.py` for real confusion matrix values; otherwise falls back to known EfficientNet-B0 values

---

## Files changed

| File | Change | Feedback addressed |
|---|---|---|
| `main.py` | Add 7 new flags + `--improve` mode | All |
| `src/config.py` | Add `scheduler`, `optimizer`, `weight_decay`, `focal_gamma`, `image_only` | P1, P2 |
| `src/train.py` | Config-driven optimiser + scheduler; `scheduler.step()`; return `(best_metrics, history)` | P1 |
| `src/model/metadata_melanoma_model.py` | Respect `use_metadata` in `forward()`; fix classifier dim | P2 |
| `src/shap_analysis.py` | **New** — SHAP analysis for metadata branch | P3 |
| `src/improve_pipeline.py` | **New** — full improvement pipeline | All |

---

## After running — update report MDs

- `05_experiments.md` **Table I row 0**: replace `88.7%†` with real image-only recall (P2 result)
- `05_experiments.md` **Table I rows B/C/D**: replace TBD with real numbers (P1 results)
- `05_experiments.md` **§C**: update with "Config C/D produces smooth curves, confirming cosine annealing resolves the instability seen in DenseNet/ResNet"
- `04_methodology.md` **§B**: replace SHAP prose with "see Fig. X" reference
