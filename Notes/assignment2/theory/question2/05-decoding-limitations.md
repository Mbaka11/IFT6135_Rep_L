# Assignment 2 — Question 2.5: limitations of decoding methods

**Goal:** identify the main limitation of greedy decoding and beam search, and connect each one to the tree in 2.4.

- [Assignment statement](../../../../Assigments/IFT6135_A26__Student_/assignment2/en/assignment2_theory_en.tex)
- [Previous: worked tree for 2.4](04-greedy-and-beam-decoding.md)
- [Earlier: beam-search complexity](03-beam-search-complexity.md)

**Status:** notes only; the LaTeX answer for 2.5 remains blank.

## 1. Greedy decoding: a local decision can discard the best sequence

Greedy decoding chooses the most likely **next token** at each step and commits to it. A locally best choice need not lead to the most likely **complete sequence**, because later conditional probabilities also contribute to the joint probability.

In the displayed tree from 2.4, greedy chooses `the red rectangle` with log-likelihood $-9.1$. But `the green rectangle` has log-likelihood

$$
-1.5-3.4-3.3=-8.2>-9.1.
$$

Greedy discards `the green` because its second edge ($-3.4$) is worse than the edge to `red` ($-3.1$). It cannot recover that branch even though the `green rectangle` continuation makes the complete sequence better. This comparison is among paths shown in the figure.

**Main limitation:** greedy decoding has no guarantee of finding the globally most likely sequence.

## 2. Beam search: pruning can discard the best continuation

Beam search keeps more alternatives, but only the best $B$ **partial sequences** at each step. A prefix ranked below the beam cutoff is discarded permanently, even if it could later lead to the best complete sequence. Thus a finite beam is still a heuristic and does **not** guarantee a global optimum.

In the standard distinct-prefix trace with $B=2$ from 2.4, the retained prefixes after step 2 are `a blue` ($-4.3$) and `a red` ($-4.5$). The algorithm discards `the green` ($-4.9$), so it cannot later select `the green rectangle` ($-8.2$). Its best displayed final candidate is `a blue square` ($-9.3$). The precise trace assumes the distinct-prefix initialization described in the [2.4 note](04-greedy-and-beam-decoding.md#5-initialization-caveat).

Increasing $B$ keeps more alternatives and costs more computation and memory. It may improve the result, but a larger finite beam does not in general guarantee the optimum or even a better final score on every problem: pruning decisions change as the beam changes. If the beam were large enough to retain **every** possible prefix, the search would be exhaustive rather than a practical bounded-beam approximation.

## 3. Side-by-side summary

| Method | What it keeps | Primary limitation |
| --- | --- | --- |
| Greedy | One locally best next token | An early local choice can rule out the best complete sequence. |
| Beam search | The $B$ highest-scoring prefixes at each step | Pruning can discard a prefix that would lead to a better complete sequence; a wider beam also costs more. |

For the short answer, the lack of a global-optimum guarantee is the key point for both methods; explain *why* separately for each. Beam search usually explores more alternatives, but it remains approximate.
