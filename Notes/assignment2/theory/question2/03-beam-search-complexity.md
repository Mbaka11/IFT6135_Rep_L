# Assignment 2 — Question 2.3: beam search and its complexity

**Goal:** understand the supplied pseudocode, count its main operations and memory, and explain what must be assumed because the top-$B$ selection implementation is not specified.

- [Assignment statement and supplied algorithm](../../../../Assigments/IFT6135_A26__Student_/assignment2/en/assignment2_theory_en.tex)
- [Previous: 2.2 — Exhaustive enumeration](02-exhaustive-enumeration.md)
- [Course slide: beam search, PDF page 48](../../../../Slides/05_RNN.pdf#page=48)

**Status:** notes only; the LaTeX answer for 2.3 remains blank. This note does not solve the language-model tree in 2.4.

## Quick reminders

- $T$ is the number of generated tokens, $N$ is the vocabulary size, and $B$ is the **beam width**: the number of partial sequences kept at each step.
- A **prefix** is a partial sequence. At each step, extend every retained prefix with every vocabulary token, then keep the $B$ highest-scoring candidates **globally**, not $B$ from each prefix.
- Logarithms turn the product of conditional probabilities into a sum. If $\ell_{t-1}^{(b)}$ is prefix $b$'s log-likelihood, extending it with token $j$ gives

$$
s_b(j)=\ell_{t-1}^{(b)}+\log p(w_t=j\mid\mathbf w_{0:t-1}^{(b)}).
$$

- A larger log-likelihood is better: for example, $-0.4$ is larger than $-2$. A zero-probability candidate has log-score $-\infty$ and cannot beat a finite score.

## 1. A small, unrelated example

Suppose $B=2$, $N=3$, and at some step two **distinct** prefixes have scores $\ell^{(1)}=-1.0$ and $\ell^{(2)}=-2.0$. Their next-token probabilities and approximate log-probabilities are:

| Prefix | $p(a),\log p(a)$ | $p(b),\log p(b)$ | $p(c),\log p(c)$ |
| --- | --- | --- | --- |
| 1 | $0.6,\;-0.511$ | $0.3,\;-1.204$ | $0.1,\;-2.303$ |
| 2 | $0.5,\;-0.693$ | $0.3,\;-1.204$ | $0.2,\;-1.609$ |

Each row's probabilities sum to $1$. Adding the prefix score gives $BN=2\cdot3=6$ candidates, with log-scores rounded to three decimals:

| Prefix | Append $a$ | Append $b$ | Append $c$ |
| --- | ---: | ---: | ---: |
| 1 | $-1.511$ | $-2.204$ | $-3.303$ |
| 2 | $-2.693$ | $-3.204$ | $-3.609$ |

The two highest are prefix 1 + $a$ ($-1.511$) and prefix 1 + $b$ ($-2.204$). Both come from the same prefix: the selection is global. These values are **not** from the assignment's 2.4 tree.

## 2. Count candidate scoring

At each of $T$ steps, the pseudocode loops over $B$ retained prefixes and $N$ tokens. It constructs

$$
B\cdot N \quad\text{candidate scores per step},
\qquad TBN \quad\text{candidate scores in total}.
$$

If each conditional log-probability is **already available**, reading it and adding the prefix score takes constant work. Candidate scoring is then $O(TBN)$. This count alone does **not** specify how the $B$ largest are found or how prefixes are copied.

A neural language model is not generally a constant-time probability oracle. If a full vocabulary distribution for one prefix at step $t$ costs $C_t$ to compute, obtaining distributions adds $O\!\left(B\sum_{t=1}^{T}C_t\right)$ work under that cost definition; avoid double-counting vocabulary computations already included in $C_t$. $C_t$ can depend on context length, model architecture, batching and caching. The question asks only in $T,B,N$, so explicitly state the simplified scoring assumption.

## 3. Selecting the top $B$: the pseudocode leaves a choice open

The instruction “find the $b$-th largest score” does not specify a sorting or selection algorithm. Let $M=BN$ candidates at one step. Different implementations give different bounds:

| Implementation | Top-$B$ selection work per step | Selection work over $T$ steps |
| --- | --- | --- |
| Sort all $M$ scores once, then take the first $B$ | $O(M\log M)$ | $O(TBN\log(BN))$ |
| Scan all $M$ scores again for each of the $B$ ranks, excluding earlier selections | $O(BM)=O(B^2N)$ | $O(TB^2N)$ |
| Maintain a min-heap of at most $B$ scores while scanning | $O(M\log(B+1))$ | $O(TBN\log(B+1))$ |

The heap may need an additional $O(B\log B)$ sort if the selected outputs must be ordered, which does not change the displayed bound for $N\ge1$. An appropriate linear-time selection algorithm can find the top $B$ in $O(M)$ per step; ordering them separately can add $O(B\log B)$.

**Why does this matter?** The three nested scoring loops do show $TBN$ score computations, but saying the **whole algorithm** is $O(TBN)$ requires an efficient selection assumption (and a strategy for prefix storage). The supplied pseudocode alone does not determine a unique selection cost. A straightforward, easy-to-explain submission can use **sort once per step** and state that assumption.

## 4. Prefix copying and space

If each retained sequence is stored as an explicit token list, at step $t$ copying its length-$(t-1)$ prefix for each of the $B$ selected candidates costs $O(Bt)$. Summed over all steps, this is $O(BT^2)$ work. Keeping old and new beams at the same time changes constants, not the order.

At a single step, materializing the candidate-score table uses $O(BN)$ space, and storing the $B$ current prefixes of length up to $T$ uses $O(BT)$. Thus an explicit implementation uses

$$
O\bigl(BN+BT\bigr)\quad\text{working/output space},
$$

excluding the model's parameters, activations and inference cache. This **sums** the score table and prefix storage; $O(BN)$ alone omits the sequences. Backpointers can avoid repeatedly copying whole prefixes, but storing the ancestry for reconstruction still has its own space cost.

Combining the pieces for **one concrete implementation**—unit-cost access to conditional log-probabilities, sorting once per step, and explicit prefix copying—gives

$$
\text{time }O\!\left(TBN\log(BN)+BT^2\right),
\qquad
\text{space }O(BN+BT),
$$

plus the model-evaluation cost described above. The scoring $O(TBN)$ is included in the sort bound for $BN\ge2$; in the degenerate one-candidate case it still costs $O(T)$. If the course expects a simplified $O(TBN)$ answer, clarify its assumptions rather than treating the unspecified selection and copying as free.

## 5. A subtle issue in the supplied initialization

The algorithm initializes **all $B$ beams** to the same `<BOS>` prefix with the same score. If duplicate hypotheses are allowed and ties are resolved by candidate index, the first selection can keep duplicate copies of the same token. As written, the pseudocode does not guarantee $B$ **distinct** sequences.

A common implementation instead begins with **one** active `<BOS>` prefix and expands it at the first step; it retains up to $B$ distinct first-token candidates, provided enough valid tokens exist. Alternatively, explicit deduplication requires its own policy. Do not silently claim the supplied initialization already guarantees diversity. The bounds above still describe its $B$-by-$N$ loops.

## 6. Is beam search practical for a large vocabulary?

It avoids enumerating all $N^T$ complete continuations by keeping only $B$ prefixes. But each step still considers up to $BN$ candidates, and selection and model inference add cost, so a very large $N$ or $B$ can still be expensive. It is often practical with a modest beam width and optimized/batched model evaluation; it is not free, and it does not guarantee the globally best sequence.

## What to retain for the short answer

State $T$ rounds and $BN$ candidates per round; give a clear **selection, prefix-storage and model-cost assumption** before stating the resulting time/space bounds. Compare with $N^T$ enumeration and qualify practicality for large vocabularies.
