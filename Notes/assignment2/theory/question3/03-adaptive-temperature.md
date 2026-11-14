# Assignment 2 — Question 3.3: adaptive temperature

**Goal:** compare a shared temperature with a temperature predicted at each decoder step, and explain why temperature alone does not reveal how focused attention is.

- [Assignment statement](../../../../Assigments/IFT6135_A26__Student_/assignment2/en/assignment2_theory_en.tex)
- [Previous: 3.2 — Scaling alignment scores](02-scaling-scores.md)
- [3.1 — Effect of temperature](01-effect-of-temperature.md)

**Status:** notes only; the LaTeX answer for 3.3 remains blank.

## 1. The two designs

At decoder step $i$, attention uses scores $e_{ij}$ and a positive temperature $\tau_i$:

$$
\alpha_{ij}=\frac{\exp(e_{ij}/\tau_i)}{\sum_k\exp(e_{ik}/\tau_i)}.
$$

- **Design A:** $\tau_i=\tau$. One learned temperature is shared across decoder steps.
- **Design B:** $\tau_i=f(s_{i-1})>0$. The temperature is predicted from the previous decoder hidden state and can vary by step.

The decoder state $s_{i-1}$ summarizes information available before step $i$. The design requires $f$ to produce positive values; for example, an implementation could use a positive-valued output function. The question does not require choosing an implementation.

## 2. Part (a): limitation and potential advantage

**Limitation of A:** its single temperature cannot independently adjust the *temperature parameter* to the needs of different decoder steps. A fixed temperature may be too restrictive if some steps benefit from sharper attention and others from broader attention.

**Potential advantage of B:** because $f$ uses the decoder state, it can adapt the temperature to the current decoding context. For example, it could predict a lower temperature at one step and a higher one at another, allowing additional flexibility in how scores are converted into weights.

This does **not** mean that A must produce identical attention distributions at every step: its alignment scores $e_{ij}$ can change with $i$, so its weights can still vary. Nor is B guaranteed to learn a useful or better temperature; the adaptive mechanism only provides the capacity to do so.

## 3. Part (b): temperature must be compared with score gaps

For two input positions $j$ and $k$ at the same decoder step,

$$
\frac{\alpha_{ij}}{\alpha_{ik}}
=\exp\!\left(\frac{e_{ij}-e_{ik}}{\tau_i}\right).
$$

The relative attention depends on the **score difference divided by temperature**, not on $\tau_i$ alone. As shown in [3.2](02-scaling-scores.md), scaling the scores and temperature together by the same positive factor leaves attention unchanged.

### A small temperature can still give broad attention

If the scores are tied, the weights are uniform for any positive temperature. Even without an exact tie, at $\tau_i=0.2$ and scores $[0.01,0]$,

$$
\alpha_i=\operatorname{softmax}([0.05,0])
\approx[0.5125,0.4875].
$$

This is nearly uniform, despite the small temperature.

### A large temperature can still give focused attention

At $\tau_i=10$ and scores $[100,0]$,

$$
\alpha_i=\operatorname{softmax}([10,0])
\approx[0.99995,0.00005].
$$

The score gap is large relative to the temperature, so the first state dominates despite the larger temperature.

**What to inspect instead:** the attention weights themselves, or score differences relative to the predicted temperature. A low or high temperature has a sharpening or flattening effect **when scores are held fixed**, but its absolute value alone does not determine selectivity.

## What to retain for a concise answer

For (a), contrast one shared temperature with a state-dependent temperature, while noting that fixed temperature does not mean fixed weights. For (b), use the score-gap ratio or a tied-score example to show why the claimed interpretation does not follow from $\tau_i$ alone.
