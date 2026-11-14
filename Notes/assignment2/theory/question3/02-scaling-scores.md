# Assignment 2 — Question 3.2: scaling alignment scores

**Goal:** see why attention depends on score differences relative to temperature, and why a learned temperature alone does not determine selectivity.

- [Assignment statement](../../../../Assigments/IFT6135_A26__Student_/assignment2/en/assignment2_theory_en.tex)
- [Previous: 3.1 — Effect of temperature](01-effect-of-temperature.md)

**Status:** notes only; the Question 3 LaTeX answer is blank. This note covers 3.2(a)–(b), not 3.3.

## 1. Compare the softmax arguments

At a fixed decoder step, let the original and scaled scores be

$$
e'_j=se_j,\qquad s>0.
$$

The original softmax uses $e_j/\tau$, whereas the new one uses $se_j/\tau'$. If we choose

$$
\boxed{\tau'=s\tau},
$$

then for every $j$,

$$
\frac{e'_j}{\tau'}=\frac{se_j}{s\tau}=\frac{e_j}{\tau}.
$$

Every numerator and the denominator are identical, so $\alpha'_j(\tau')=\alpha_j(\tau)$ for every $j$. This relationship works for **any** score vector, not just one particular example.

If all scores happen to be equal (or there is just one state), both distributions are uniform for every positive temperature, so other temperature choices also work for that particular vector. With at least two distinct scores, $\tau'=s\tau$ is necessary as well as sufficient: the probability ratio below fixes the scale.

## 2. Ratios show what matters

For any pair $j,k$,

$$
\frac{\alpha_j}{\alpha_k}
=\exp\left(\frac{e_j-e_k}{\tau}\right).
$$

A common offset added to every score cancels; what matters for relative weights is the **difference** $e_j-e_k$ divided by $\tau$. Multiplying both score differences and temperature by $s$ leaves this ratio, and therefore all normalized weights, unchanged.

For example, $e=[2,1,0]$ at $\tau=1$ and $e'=[0.4,0.2,0]$ at $\tau'=0.2$ have the same scaled scores $[2,1,0]$. Both give approximately

$$
\alpha=[0.6652,\;0.2447,\;0.0900].
$$

The temperature changed fivefold, but the attention distribution did not.

## 3. Part (b): is $\tau=0.2$ enough to tell?

**No.** A small temperature produces sharp attention only relative to sufficiently large **score differences**. If score scale and temperature are both learned, the model may also make the scores very close to one another.

For a concrete comparison, use two states and the same temperature $\tau=0.2$:

| Scores | Scores divided by $\tau$ | Attention weights (approximately) | Interpretation |
| --- | --- | --- | --- |
| $[0.01,0]$ | $[0.05,0]$ | $[0.5125,0.4875]$ | Nearly uniform |
| $[2,0]$ | $[10,0]$ | $[0.99995,0.00005]$ | Highly selective |

Same temperature, very different attention. If the scores are exactly tied, the weights are uniform regardless of positive temperature. Conversely, a larger temperature can still give sharp attention if score gaps are large enough.

The learned temperature and global score scale are **not separately identifiable from the attention weights alone**: jointly scaling both by the same positive factor leaves those weights unchanged. This is a statement about this softmax distribution, not necessarily about every other part of a complete model. Inspect the actual weights or the score gaps relative to temperature before concluding how selective attention is.

## What to retain for a concise answer

For (a), substitute $e'_j=se_j$ and choose $\tau'=s\tau$. For (b), say no and give a near-tie counterexample or explain the ratio $(e_j-e_k)/\tau$.
