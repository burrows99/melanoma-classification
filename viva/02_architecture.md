# Model Architecture

← [Dataset](01_dataset_preprocessing.md) | [Index](index.md) | Next: [Focal Loss →](03_focal_loss_imbalance.md)

---

## The Architecture in One Sentence

An ImageNet-pretrained **EfficientNet-B0** extracts a 1280-d embedding from the image; a two-layer **metadata MLP** produces a 64-d embedding; these are **concatenated to 1344-d** and projected to a single logit via a linear head, trained end-to-end with sigmoid focal loss.

---

## Grilling Question 1: "Why EfficientNet-B0? Why not ResNet-50 or DenseNet-121?"

| Model | Params | Top-1 ImageNet | Notes |
|-------|--------|---------------|-------|
| ResNet-50 | 25.6M | 76.1% | More params for same accuracy; simple depth scaling |
| DenseNet-121 | 8.0M | 74.4% | Dense connections add memory overhead |
| **EfficientNet-B0** | **5.3M** | **77.1%** | Compound scaling; best accuracy-per-FLOP |

---

### Before reading: universal concepts

Every CNN does the same job: take a pixel image → produce a flat list of numbers that summarises "what's in this image." They all do this by:

1. **Sliding a small filter (window) across the image** — the filter fires a high number when it detects the pattern it was trained for (e.g. a diagonal edge). This is a *convolution*.
2. **Stacking many filters** — so instead of 3 colour numbers per pixel, you get e.g. 64 "detector" numbers per pixel location.
3. **Progressively shrinking the image** (via stride/pooling) while adding more detectors — early layers see fine pixel detail, later layers see large abstract patterns.
4. **Collapsing to a vector** at the end (GlobalAvgPool) — average each detector across the whole image → one number per detector → flat vector.

The three architectures differ in **how they connect the layers** — and that's the key choice.

---

### The Core Difference: How Each Architecture Connects Its Layers

```
INPUT IMAGE (256×256 pixels, 3 colour channels = RGB)
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE          ResNet-50              DenseNet-121         EfficientNet-B0 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  How layers      Each block gets       Each layer gets      Each block gets │
│  are connected:  only the PREVIOUS     ALL previous         only PREVIOUS   │
│                  block's output        layers' outputs      block's output  │
│                  + adds original       concatenated         + SE attention  │
│                  input back (skip)     together (pile)      to re-weight    │
│                                                             channels        │
│                  output = F(x) + x     output = [x₀,x₁…xₙ] output = SE(F(x)) + x│
│                                                                             │
│  WHY:            Lets gradients        Early edge           Learns WHICH    │
│                  flow freely to        detectors never      detectors       │
│                  early layers          get forgotten        actually matter │
│                  (avoids vanishing     (always visible      for THIS image  │
│                  gradient)             to later layers)     (attention)     │
├─────────────────────────────────────────────────────────────────────────────┤
│  STEM            256→64×64            256→64×64            256→128×128     │
│  (first pass     Big 7×7 filter       Big 7×7 filter       Smaller 3×3     │
│   of image)      halves size twice    halves size twice    filter, halves  │
│                  → coarse features    → coarse features    once only       │
│                                                                             │
│  WHY ResNet/Dense use big 7×7: designed before efficient small-filter      │
│  techniques existed. EfficientNet uses 3×3 — cheaper, works just as well.  │
├─────────────────────────────────────────────────────────────────────────────┤
│  EARLY STAGES    3 blocks             Dense Block 1        Stages 1-3      │
│  (fine detail:   Each block:          (6 layers):          Each MBConv:    │
│  edges, skin     compress→3×3→expand  Each new layer       expand×6 →      │
│  texture)        + skip add           stacks its output    depthwise 3×3 → │
│                                       onto the pile        SE attention →  │
│                  64×64, 256 detect.   64×64, 256 channels  compress back   │
│                                                                             │
│                  Skip connection      Pile grows by 32     SE asks: "which │
│  WHY:            means if this        channels per layer   of my 256       │
│                  block learns         — early edge         channels are    │
│                  nothing useful,      detectors stay       relevant to     │
│                  it passes input      accessible forever   this patch?"    │
│                  through unchanged    (never overwritten)  and turns down  │
│                                                            the others      │
├─────────────────────────────────────────────────────────────────────────────┤
│  MID STAGES      Stages 2-3           Transitions +        Stages 4-5      │
│  (bigger         Image halved at      Dense Blocks 2-3:    Spatial size    │
│  patterns:       each stage           Compress pile,       stays at        │
│  shapes,         32×32→16×16          then grow again      14×14 for 2     │
│  borders)        512→1024 detect.     32×32→16×16          stages          │
│                                       512→1024 channels    80→112 channels │
│                                                                             │
│  WHY halve the image? A 256px lesion border detail isn't needed once you   │
│  know "there IS a border here." Smaller spatial → faster, more abstract.   │
│                                                                             │
│  WHY DenseNet compresses: after 12 layers of +32 channels each, the pile   │
│  is ~512 wide. Without compression, memory grows quadratically.            │
├─────────────────────────────────────────────────────────────────────────────┤
│  LATE STAGES     Stage 4              Dense Blocks 3-4     Stages 6-7      │
│  (semantic:      3 blocks             24+16 layers         4+1 blocks      │
│  "irregular      8×8, 2048 detect.    8×8, 1024 channels   7×7, 320 ch.    │
│  lesion")                                                                   │
│                                                                             │
│  By this point every network is asking:                                     │
│  "Does this image contain an irregular, asymmetric, dark-bordered lesion?" │
├─────────────────────────────────────────────────────────────────────────────┤
│  FINAL HEAD      Conv 1×1             GlobalAvgPool        Conv 1×1 →      │
│  (collapse to    GlobalAvgPool        → 1024-d vector      7×7×1280        │
│  a vector)       → 2048-d vector                           GlobalAvgPool   │
│                                                            → 1280-d vector │
│                                                                             │
│  Linear(2048→1000) [removed]  Linear(1024→1000)[removed]  Linear(1280→1000)[removed]│
│                                                                             │
│  YOU EXTRACT:    2048-d              1024-d               1280-d           │
│                  (richest but        (lowest, yet         (best balance:   │
│                   25.6M params)       8M params)           5.3M params)    │
└─────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
OUTPUT VECTOR fed into your fusion head
```

---

### Why the Skip/Dense/SE differences actually matter for training

| Problem | What causes it | ResNet fix | DenseNet fix | EfficientNet fix |
|---------|---------------|-----------|-------------|-----------------|
| Gradients vanish in deep nets | Chain rule: multiply many numbers <1 → 0 | Skip adds +x, gradient always has a path back | Same skip logic via concatenation | Skip + SE; SE also avoids wasting gradient on irrelevant channels |
| Early features overwritten | Later layers overwrite what earlier learned | Skip preserves some, but mostly overwrites | Concatenation: old features *never* overwritten | SE: network explicitly votes to keep or suppress old channels |
| Model wastes compute on easy patterns | No mechanism to say "ignore this" | None | None | SE attention: suppress channels that don't help for THIS image |
| Too many parameters | Deep/wide networks bloat | 25.6M (unavoidable with standard convs) | 8M (dense reuse is efficient) | 5.3M (depthwise separable convolutions are ~9× cheaper) |

---

### The Depthwise vs Standard Convolution Difference (why EfficientNet is so cheap)

**Standard convolution** (used by ResNet and DenseNet):  
One filter looks at ALL channels simultaneously at each location.  
64 input channels × 64 output channels × 3×3 filter = **36,864 multiplications per pixel location**.

**Depthwise separable convolution** (used by EfficientNet):  
Step 1 — each channel gets its own 3×3 filter (spatial only): 64 × 3×3 = **576 multiplications**  
Step 2 — mix channels with 1×1 filter: 64 × 64 = **4,096 multiplications**  
Total: **4,672 multiplications** — about **8× cheaper** for the same output shape.

This is why EfficientNet achieves higher accuracy with far fewer parameters.

---

### Compound Scaling — What Makes B0→B7 Possible

$$
\text{depth} = \alpha^\phi, \quad \text{width} = \beta^\phi, \quad \text{resolution} = \gamma^\phi
$$

where $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ (FLOPs roughly double per unit φ). B0 is the base (φ=1). For B0: α=1.2, β=1.1, γ=1.15.

| Scale alone | Problem |
|-------------|---------|
| More depth only (ResNet-style) | Resolution stays low — later layers can't see fine detail |
| More width only | No more layers to build abstract concepts |
| Higher resolution only | More pixels but same shallow network — slow, no benefit |
| **All three together (EfficientNet)** | Each axis compensates for the others' limitations |

**Domain justification:** The ISIC 2020 *winning solution* used an EfficientNet ensemble (AUC=0.949). Its smooth convergence curve also made it well-suited for controlled HP studies — you can attribute metric changes to the HP, not to noisy loss landscapes.

---

## Grilling Question 2: "Describe the dual-branch architecture exactly."

**Branch 1 — Image branch:**
- EfficientNet-B0 with the final classification head *removed*
- Outputs a **1280-d feature vector** (the penultimate embedding layer)
- Pretrained on ImageNet; fine-tuned end-to-end

**Branch 2 — Metadata MLP:**
- Input: 14-d one-hot/z-scored vector
- Layer 1: Linear(14 → 128) + BatchNorm + ReLU + Dropout(0.3)
- Layer 2: Linear(128 → 64) + BatchNorm + ReLU + Dropout(0.3)
- Output: **64-d embedding**

**Fusion:**
- Concatenate: [1280-d ‖ 64-d] = **1344-d**
- Linear(1344 → 1) → single logit
- Sigmoid activation → probability of malignancy

**Training:** End-to-end. Both branches receive gradients from the focal loss simultaneously.

---

## Grilling Question 3: "Why *late* fusion? Why not early or mid fusion?"

**Early fusion** would concatenate raw metadata directly to the image pixels — meaningless, as a patient's age has nothing to do with individual pixel values.

**Mid fusion** would concatenate metadata into an intermediate CNN feature map layer — forces the metadata to interact with spatial features at a fixed resolution, which doesn't make intuitive sense.

**Late fusion** lets each branch develop its own representation in its own "language":
- The CNN learns spatial, textural, and colour features from pixels
- The MLP learns demographic and anatomical risk patterns from tabular data
- They meet only at decision time, contributing complementary signals

This mirrors clinical practice: a radiologist first reads the scan, then the nurse provides patient context — they don't look at both simultaneously from the start.

**Drill-Down:** *Did you consider gated fusion?*

Gated fusion is where a small network learns **how much to trust each source per sample**, rather than always combining them equally:

```
Your late fusion (fixed):
  image_features (1280-d) ──┐
                             concat → Linear → logit
  metadata_features (64-d) ──┘
  (always combined with equal opportunity; head learns fixed weights)

Gated fusion (dynamic):
  image_features   ──┐
                      ├→ gate network → g ∈ [0,1]
  metadata_features ──┘
  output = g × image_features + (1-g) × metadata_features
  (g learned per sample: clear image → trust image; ambiguous → trust metadata)
```

**Why fixed concatenation is sufficient here:**
- Gated fusion is designed for cases where the two modalities are sometimes *contradictory* (e.g. image says benign, metadata says high-risk patient). Your modalities are orthogonal — they carry complementary, not competing, information.
- SHAP values are stable and consistent across all five experiments — metadata contribution doesn't vary wildly per sample, so a fixed linear combination is appropriate.
- It adds a new learnable gate network and another hyperparameter to tune — adding complexity without clear gain given the ablation evidence.
- Fixed concatenation (Pacheco et al., 2020) is the standard baseline for image+tabular fusion in medical imaging.

Gated fusion is a valid future direction if you ever observe that metadata *hurts* certain image subtypes.

**Drill-Down:** *What is the parameter cost of the metadata branch?*
- Layer 1: 14×128 + 128 = 1,920 params
- Layer 2: 128×64 + 64 = 8,256 params
- Fusion head: 1344×1 + 1 = 1,345 params
- **Total metadata overhead: ~11.5k params** vs 5.3M for the backbone — negligible.

---

## Grilling Question 4: "Why remove the EfficientNet classification head?"

EfficientNet-B0's original head classifies 1000 ImageNet classes. Removing it gives us the raw feature representation — the learned "visual vocabulary" of ImageNet. We then attach our own head that:
1. Takes both image and metadata
2. Outputs a single binary logit (not 1000 class scores)

Keeping the original head would mean throwing away the metadata branch entirely.

---

## Grilling Question 5: "Why Dropout(0.3) in the metadata MLP?"

With only 14 input features but 128 hidden units, the MLP could easily memorise training metadata patterns (e.g., "all 80-year-olds on the torso → malignant"). Dropout(0.3) randomly zeros 30% of neurons during training, forcing the network to learn redundant representations and preventing it from latching onto spurious correlations in demographic data.

---

## Layman Analogy

*"The dual-branch model is like a hospital consultation. One specialist (the CNN) spends 20 minutes reading your dermoscopy scan. A second specialist (the MLP) reviews your patient chart — your age, sex, where the mole is. Neither talks to the other until the very end, when they both sit down together and make the final call. Late fusion is that final meeting room."*

---

← [Dataset](01_dataset_preprocessing.md) | [Index](index.md) | Next: [Focal Loss →](03_focal_loss_imbalance.md)
