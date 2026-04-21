# Model Architecture

← [Dataset](01_dataset_preprocessing.md) | [Index](index.md) | Next: [Focal Loss →](03_focal_loss_imbalance.md)

---

> **Core Insight:** EfficientNet-B0 gives the best accuracy-per-parameter ratio via depthwise separable convolutions + SE attention. Late fusion lets image and metadata speak their own "languages" before meeting at decision time.

---

## The Architecture in One Sentence

An ImageNet-pretrained **EfficientNet-B0** extracts a 1280-d embedding from the image; a two-layer **metadata MLP** produces a 64-d embedding; these are **concatenated to 1344-d** and projected to a single logit via a linear head, trained end-to-end with sigmoid focal loss.

---

## Grilling Question 1: "Why EfficientNet-B0? Why not ResNet-50 or DenseNet-121?"

| Model | Params | Top-1 ImageNet | Verdict |
|-------|--------|---------------|---------|
| ResNet-50 | 25.6M | 76.1% | 4.8× more params, worse accuracy |
| DenseNet-121 | 8.0M | 74.4% | Dense connections add memory overhead |
| **EfficientNet-B0** | **5.3M** | **77.1%** | **Best accuracy-per-FLOP** |

**Why EfficientNet wins:**
1. **Depthwise separable convolutions** — ~9× cheaper than standard convs (see below)
2. **SE attention** — each block learns which channels matter for *this* image
3. **Compound scaling** — B0→B7 scales depth, width, and resolution together, not just one axis
4. **Domain proof:** ISIC 2020 winning solution used EfficientNet ensemble (AUC=0.949)

---

### Why Depthwise Separable Convolutions Are So Efficient

**Standard convolution** (ResNet, DenseNet):  
One filter looks at ALL channels at once. 64 input × 64 output × 3×3 = **36,864 multiplications**.

**Depthwise separable convolution** (EfficientNet):
- Step 1: Each channel gets its own 3×3 filter → 64 × 3×3 = **576**
- Step 2: Mix channels with 1×1 → 64 × 64 = **4,096**
- **Total: 4,672** — ~8× cheaper for same output.

---

### Compound Scaling — The B0→B7 Formula

$$\text{depth} = \alpha^\phi, \quad \text{width} = \beta^\phi, \quad \text{resolution} = \gamma^\phi$$

where $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ (FLOPs double per φ step).

| Scale alone | Problem |
|-------------|---------|
| More depth only | Resolution stays low — later layers miss fine detail |
| More width only | No more layers to build abstract concepts |
| Higher resolution only | Same shallow network — slow, no benefit |
| **All three together** | Each axis compensates for the others |

---

## Grilling Question 2: "Describe the dual-branch architecture exactly."

### What Was Chopped From EfficientNet-B0

**Original ImageNet classifier head (removed):**
```
classifier = Sequential(
    Dropout(p=0.2, inplace=True),
    Linear(in_features=1280, out_features=1000)  # 1000 ImageNet classes
)
```
- **Parameters removed:** 1280 × 1000 + 1000 = **1,281,000 params**
- **Why remove it?** ImageNet classes (dog, cat, car…) have zero overlap with melanoma classification. Keeping it would force the model to unlearn 1000-class semantics.

**Replacement:**
```
classifier = nn.Identity()  # Passthrough — outputs raw 1280-d features
```

### Branch 1 — Image Branch (Modified EfficientNet-B0)

| Stage | Layer | Output Shape | Parameters |
|-------|-------|--------------|------------|
| Input | — | 3×256×256 | 0 |
| Conv stem | Conv2d(3,32,3×3,stride=2) + BN + SiLU | 32×128×128 | 864 |
| MBConv blocks | 16 inverted residual blocks | 1280×8×8 | ~5.0M |
| Head | Conv2d + BN + SiLU + Pool | 1280 | ~1k |
| **Classifier** | **Identity()** | **1280** | **0** |

**Total backbone:** ~5.3M params (unchanged from original EfficientNet-B0)

### Branch 2 — Metadata MLP

| Layer | Operation | Input → Output | Parameters |
|-------|-----------|----------------|------------|
| Input | — | 14 | 0 |
| Linear 1 | Linear(14→128) | 128 | 14×128 + 128 = **1,920** |
| BN 1 | BatchNorm1d(128) | 128 | 128×2 = **256** |
| Act 1 | ReLU | 128 | 0 |
| Drop 1 | Dropout(0.3) | 128 | 0 |
| Linear 2 | Linear(128→64) | 64 | 128×64 + 64 = **8,256** |
| BN 2 | BatchNorm1d(64) | 64 | 64×2 = **128** |
| Act 2 | ReLU | 64 | 0 |
| Drop 2 | Dropout(0.3) | 64 | 0 |

**Total MLP:** 1,920 + 256 + 8,256 + 128 = **10,560 params**

### Fusion Head

| Layer | Input → Output | Parameters |
|-------|----------------|------------|
| Concat | [1280 ‖ 64] = 1344 | 0 |
| Linear | 1344 → 1 | 1344×1 + 1 = **1,345** |

### Full Model Parameter Summary

| Component | Parameters | % of Total |
|-----------|------------|------------|
| EfficientNet-B0 backbone | 5,285,286 | 99.78% |
| Metadata MLP | 10,560 | 0.20% |
| Fusion head | 1,345 | 0.03% |
| **Total** | **5,297,191** | 100% |

**Key insight:** The metadata branch and fusion head add only **0.23%** overhead — essentially free multimodal learning.

---

## Grilling Question 3: "Why *late* fusion? Why not early or mid fusion?"

| Fusion Type | What it does | Problem |
|-------------|--------------|---------|
| **Early** | Concatenate metadata to image pixels | Age has no spatial meaning in a pixel grid |
| **Mid** | Inject metadata into CNN feature maps | Forces metadata to interact at fixed resolution — arbitrary |
| **Late** | Each branch develops independently, meets at decision | Each modality speaks its own "language" |

**Clinical analogy:** A radiologist reads the scan first, then the nurse provides patient context at decision time — not simultaneously from the start.

**Mathematical justification:** Under conditional independence (image features ⊥ metadata features | class), late fusion is Bayes-optimal:
$$P(\text{malignant} | \mathbf{x}_\text{img}, \mathbf{x}_\text{meta}) \propto P(\mathbf{x}_\text{img} | \text{malignant}) \cdot P(\mathbf{x}_\text{meta} | \text{malignant})$$

**Drill-Down:** *Did you consider gated fusion?*  
Gated fusion learns a dynamic weight `g ∈ [0,1]` per sample to trust image vs metadata more. Here, SHAP shows metadata contribution is stable across samples — a fixed linear combination is sufficient. Gated fusion adds complexity without clear gain.

---

## Grilling Question 4: "Why Dropout(0.3) in the metadata MLP?"

With 14 input features but 128 hidden units, the MLP could memorise patterns like "all 80-year-olds on torso → malignant." Dropout(0.3) forces the network to learn redundant representations, preventing spurious demographic correlations.

---

## Layman Analogy

*"The dual-branch model is a hospital consultation. One specialist (CNN) reads your scan. Another (MLP) reviews your chart. Neither talks until the final meeting, where both contribute their expertise. Late fusion is that meeting room."*

*"EfficientNet vs ResNet is a fuel-efficient sports car vs a gas-guzzling truck. The truck is bigger, but the sports car is faster and cheaper to run. The EfficientNet 'family' scales like race teams building bigger cars from the same blueprint."*

---

← [Dataset](01_dataset_preprocessing.md) | [Index](index.md) | Next: [Focal Loss →](03_focal_loss_imbalance.md)