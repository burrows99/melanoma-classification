# Optimizers & Learning Rate Schedulers

← [Experiments](04_experiments.md) | [Index](index.md) | Next: [Evaluation →](06_evaluation_metrics.md)

---

## What Does an Optimizer Even Do?

Imagine the model has millions of knobs (weights). At the start, they're all random — the model is useless. Training is the process of turning those knobs until the model makes good predictions.

**The optimizer is the thing that turns the knobs.**

Here's the loop:
1. Feed a batch of images through the model → get predictions
2. Compute the loss (how wrong the predictions were)
3. Backpropagation calculates, for each knob: *"if I turn this knob slightly, does the loss go up or down?"* — this is the **gradient**
4. The optimizer uses those gradients to decide how much to turn each knob

The simplest possible optimizer (SGD) just says: *"turn every knob a little bit in the direction that reduces loss."* The learning rate controls how big each turn is.

**Why not just use SGD?** Some knobs matter a lot, some barely at all. SGD turns them all the same amount, which is wasteful. Adam is smarter — it tracks each knob's *history* and turns frequently-updated knobs less (they're already tuned) and rarely-updated knobs more.

**So what's AdamW?** Adam has one problem: after thousands of updates, some knobs get turned to extreme positions — very large positive or negative values. A knob at an extreme position is over-confident: it's saying *"this one feature is overwhelmingly important, ignore everything else."* That's usually overfitting.

**Weight decay** is the fix: after every update, nudge every knob slightly back toward zero — like a gentle spring pulling it to centre. This stops any single knob from dominating.

**The catch:** In standard Adam, this spring force gets mixed into the gradient before Adam's adaptive scaling divides it. So the spring is stronger on some knobs than others depending on gradient history — not what you want.

**AdamW** separates the two steps: first do the Adam update, *then* apply the spring. Every knob gets an equally strong pull back toward centre, regardless of its gradient history. That's the "decoupled" part.

---

## The Contenders

| | Adam | AdamW |
|---|------|-------|
| Update rule | Adaptive moment estimation | Adam + decoupled weight decay |
| Weight decay | Folded into gradient (L2 reg) | Applied directly to weights (correct L2) |
| Effect on minority class | Neutral | Can smooth sharp minority-class boundaries |
| Used in | Baseline, Exp 1, Exp 4 | Exp 2, Exp 3 |

---

## Grilling Question 1: "What is Adam doing mathematically?"

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(1st moment: mean)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(2nd moment: variance)}$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \quad \text{(bias correction)}$$
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**In plain English:** Adam maintains a running average of the gradient and its squared magnitude. Parameters with historically large gradients get smaller updates (adaptive). This makes it well-suited for noisy, sparse gradients — exactly what you get from a class-imbalanced dataset where malignant gradients are rare but large.

**Hyperparameters used:** LR=10⁻⁴, β₁=0.9, β₂=0.999 (PyTorch defaults).

---

## Grilling Question 2: "What's wrong with L2 regularisation in standard Adam?"

Recall the spring analogy: weight decay is a spring pulling every knob back toward zero after each update, to prevent any single weight from becoming extreme.

In standard Adam with L2 regularisation, the spring force is just **added to the gradient** before the update:

$$g_t^{\text{reg}} = g_t + \lambda \theta_{t-1}$$

Then the whole combined signal gets divided by Adam's adaptive scaling factor $\sqrt{\hat{v}_t}$.

**The bug:** that scaling factor is based on *gradient history* — parameters that received large gradients frequently get divided by a big number (small effective update). But now the spring force is in that same signal, so it also gets divided. The spring ends up weaker on the most active parameters and stronger on the inactive ones. **The decay is unequal — it depends on gradient history, not just weight magnitude.**

**AdamW** separates the two steps cleanly:

$$\theta_t = \underbrace{(1 - \eta\lambda)\theta_{t-1}}_{\text{spring: always proportional to weight}} - \underbrace{\frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t}_{\text{Adam gradient step}}$$

Spring first, Adam step second — they never mix. Every weight gets the same proportional pull toward zero ($\lambda$ times its current value), regardless of gradient history. This is "**decoupled** weight decay."

---

## Grilling Question 3: "So why did AdamW make things worse in Exp 2?"

AdamW's correct weight decay penalises large weights uniformly. In a standard balanced classification task, this prevents overfitting by shrinking dominant parameters.

**The problem in Exp 2:** Detecting melanoma (the minority class) relies on a small number of specialised neurons that have learned to fire strongly for rare malignant features (asymmetric borders, atypical pigmentation). AdamW shrinks these large weights, effectively **eroding the sharp decision boundary** the model had built around the minority class.

The result: FP surges 145 → 181. The model becomes more cautious about predicting benign — it loses certainty about what "safe" benign looks like, and flags more borderline cases as malignant.

**Ironically, recall improved** (93.54% — highest of all experiments) because the looser boundary catches more true malignant cases. But the clinical cost of 36 extra false surgeries outweighed the 4 extra catches.

---

## Grilling Question 4: "Explain cosine annealing."

**Fixed LR (Baseline, Exp 4):** Learning rate stays constant at 10⁻⁴ throughout all 20 epochs. The model keeps taking the same size steps regardless of whether it's near a minimum.

**Cosine Annealing (Exp 1–3):**
$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{t\pi}{T}\right)$$

Starting at LR=10⁻⁴, the learning rate decreases following a cosine curve toward a minimum (e.g., 0 or a small floor), then may restart (cosine annealing with warm restarts / SGDR).

**Why it works:** Early epochs take large steps to escape poor local minima. As training progresses, the LR decreases smoothly, allowing the model to settle into the bottom of a sharp, narrow minimum — rather than bouncing around it with a constant large step. Sharp minima tend to generalise better on the training-distribution data.

**Effect on FP:** Cosine annealing drove training loss lower (0.0152→0.0054 converging tighter) but widened the train/val gap slightly (0.36–0.59%). Best-checkpoint selection by val F1 mitigates this.

---

## Grilling Question 5: "Why LR=10⁻⁴ specifically?"

This is a standard fine-tuning learning rate for pre-trained ImageNet models. The backbone's weights are already good — you don't want to destroy them with a large LR (catastrophic forgetting). 10⁻⁴ is small enough to refine the features gently while adapting to the dermoscopy domain. It's also used in the ISIC 2020 winning solution.

---

## Layman Analogy

*"Adam is a hiker who adjusts their stride based on the terrain — short steps uphill, longer steps downhill. AdamW is the same hiker but wearing a heavy backpack (weight decay) that makes every step slightly harder, regardless of terrain — designed to stop the hiker from taking wild leaps. The problem is, in a minority-class landscape where a few narrow canyons represent malignant features, the backpack makes it harder to stand precisely inside the canyon, so the hiker slips out (FP spike).*

*Cosine annealing is the hiker gradually slowing down as they approach their destination. Large confident strides at first, gentle careful steps at the end — so they park exactly at the lowest point rather than overshooting it."*

---

← [Experiments](04_experiments.md) | [Index](index.md) | Next: [Evaluation →](06_evaluation_metrics.md)
