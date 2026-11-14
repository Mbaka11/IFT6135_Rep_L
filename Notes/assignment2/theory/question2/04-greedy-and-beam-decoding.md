# Assignment 2 — Question 2.4: greedy and beam decoding

**Goal:** follow the given tree step by step, keeping track of both the next-edge score and the cumulative sequence score.

- [Assignment statement and tree](../../../../Assigments/IFT6135_A26__Student_/assignment2/en/assignment2_theory_en.tex)
- [Previous: beam-search complexity](03-beam-search-complexity.md)
- [Next: limitations of decoding](05-decoding-limitations.md)

**Status:** notes only; the LaTeX answers for 2.4 remain blank.

## 1. Reading the tree

Each edge is labeled with a **conditional log-probability**. To find the log-likelihood of a whole sequence, add its edge labels:

$$
\log p(w_{1:3}\mid w_0)
=\log p(w_1\mid w_0)+\log p(w_2\mid w_{0:1})+\log p(w_3\mid w_{0:2}).
$$

For example, `the green rectangle` has score $-1.5-3.4-3.3=-8.2$. Larger log-scores are better: $-8.2>-9.3$.

The statement says that low-probability paths are omitted from the drawing. The calculations below use the displayed tree; without the omitted scores, comparisons are explicitly among the displayed paths.

## 2. Part (a): greedy decoding

Greedy decoding chooses the best **next edge from the current node**, then never revisits that decision.

| Step | Current prefix | Edge choices shown | Choice | Cumulative score |
| --- | --- | --- | --- | ---: |
| 1 | `<BOS>` | `the`: $-1.5$; `a`: $-1.8$ | `the` | $-1.5$ |
| 2 | `the` | `green`: $-3.4$; `red`: $-3.1$ | `red` | $-4.6$ |
| 3 | `the red` | `circle`: $-4.8$; `rectangle`: $-4.5$ | `rectangle` | $-9.1$ |

Thus the greedy sequence is **`the red rectangle`**, with log-likelihood

$$
-1.5-3.1-4.5=\boxed{-9.1}.
$$

The greedy choice compares edges from the *same* prefix. The cumulative total is kept to report the final log-likelihood, not to compare unrelated prefixes at the next greedy step.

## 3. Part (b): beam search with $B=2$

At each step, extend **every retained prefix**, add the edge log-probability to its cumulative score, and keep the two highest cumulative scores **across all candidates**. In the following trace we use the standard distinct-prefix convention: start with one active `<BOS>` and expand it. The supplied pseudocode initializes two identical BOS entries; without a deduplication or tie-breaking rule, it can instead retain duplicate hypotheses. See the caveat below.

### Step 1

| Candidate | Cumulative score | Keep? |
| --- | ---: | --- |
| `the` | $-1.5$ | Yes |
| `a` | $-1.8$ | Yes |

The beam is `[the, a]`.

### Step 2

| Candidate | Calculation | Cumulative score | Keep? |
| --- | --- | ---: | --- |
| `the green` | $-1.5-3.4$ | $-4.9$ | No |
| `the red` | $-1.5-3.1$ | $-4.6$ | No |
| `a blue` | $-1.8-2.5$ | $-4.3$ | Yes |
| `a red` | $-1.8-2.7$ | $-4.5$ | Yes |

The beam is `[a blue, a red]`. Notice that **both** survivors come from `a`: keeping one child per parent would not be the stated global top-two rule. The entire `the` branch is pruned, including the sequence selected by greedy decoding.

### Step 3

| Candidate | Calculation | Cumulative score | Keep? |
| --- | --- | ---: | --- |
| `a blue ellipse` | $-4.3-5.4$ | $-9.7$ | No |
| `a blue square` | $-4.3-5.0$ | $-9.3$ | Yes |
| `a red circle` | $-4.5-4.9$ | $-9.4$ | Yes |
| `a red square` | $-4.5-5.1$ | $-9.6$ | No |

The final beam is:

1. **`a blue square`**, log-likelihood $\boxed{-9.3}$.
2. **`a red circle`**, log-likelihood $\boxed{-9.4}$.

## 4. Why can greedy beat beam search here?

The beam's best final score, $-9.3$, is worse than greedy's $-9.1$. That is possible because beam search ranks **partial** sequences, not their unknown final scores. At step 2 it discards `the red` ($-4.6$) in favor of `a blue` ($-4.3$) and `a red` ($-4.5$), but their later edges have worse scores.

In fact, `the green rectangle` scores $-8.2$, so it is better than either of those outputs among the displayed paths. Greedy discards it at step 2, while width-two beam search also discards it. Neither method is guaranteed to find the best complete sequence.

## 5. Initialization caveat

Algorithm 2 sets both initial beams to the same `<BOS>` with score zero. The first expansion therefore contains duplicate candidate entries. If duplicates are allowed, the result depends on tie-breaking and need not match the distinct-prefix trace above. For example, selecting both copies of `the` first can retain duplicate continuations. Starting with one active prefix (or deduplicating candidates) gives the usual distinct-sequence interpretation used in the trace; state this assumption if presenting the result.
