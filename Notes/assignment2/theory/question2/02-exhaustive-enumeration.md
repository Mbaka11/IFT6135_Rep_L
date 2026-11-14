# Assignment 2 — Question 2.2: exhaustive enumeration

**Goal:** explain why trying every possible continuation quickly becomes impractical as sequence length or vocabulary size grows.

- [Assignment statement](../../../../Assigments/IFT6135_A26__Student_/assignment2/en/assignment2_theory_en.tex)
- [Previous: 2.1 — Counting sequences](01-counting-sequences.md)
- [Next: 2.3 — Beam search](03-beam-search-complexity.md)

**Status:** notes only; the LaTeX answer for 2.2 remains blank.

## Quick reminders

- A sequence's joint probability multiplies the conditional probabilities of its tokens:

$$
p(w_{1:T}\mid w_0)=\prod_{t=1}^{T}p(w_t\mid w_0,\ldots,w_{t-1}).
$$

- **Exhaustive enumeration** means considering every allowed length-$T$ continuation and choosing the one with the highest joint probability. Under the unrestricted counting assumption in 2.1, there are $N^T$ candidates.
- The model uses the **full prefix**: the probability of a word can depend on the entire preceding sequence, not just on the previous word. A locally likely next word does not by itself guarantee the most likely complete sequence.

## 1. See the growth with a small example

With $N=3$ choices per generated position:

| Generated length $T$ | Full sequences $N^T$ |
| ---: | ---: |
| 1 | 3 |
| 2 | 9 |
| 3 | 27 |
| 10 | 59,049 |

Each additional position triples the number of complete candidates. The fixed `<BOS>` is not another generated choice. For a larger example, $N=10{,}000$ and $T=10$ give

$$
N^T=(10^4)^{10}=10^{40}
$$

possible continuations. This is the number of candidates alone, before accounting for the cost of evaluating their probabilities.

## 2. What does enumeration require?

A straightforward method generates each continuation, evaluates its joint probability using the conditional factors, and keeps the best. If it recomputes all $T$ factors for each candidate, it performs $O(TN^T)$ **conditional-score accesses**, assuming each access costs one unit. The language-model calculation itself can be much more expensive; the assignment does not specify its cost.

Prefixes can be reused: for example, the same probability for $w_1=a$ is shared by all continuations starting with $a$. A tree traversal that evaluates each prefix edge once examines

$$
N+N^2+\cdots+N^T
=\frac{N^{T+1}-N}{N-1}=O(N^T)\quad(N>1).
$$

Prefix sharing removes repeated calculations, but the number of leaves is still $N^T$: **it does not make exhaustive search scalable**. Keeping only the current best candidate can save memory, but it does not eliminate the search work.

## 3. Answer the scalability question carefully

For fixed $N>1$, the number of candidates is **exponential in $T$**. For fixed $T$, it is a degree-$T$ polynomial in vocabulary size $N$, not literally exponential in $N$. In either case, for realistic values of both $N$ and $T$, enumerating $N^T$ sequences is usually infeasible.

An exhaustive search can be practical for very small vocabularies and short sequences. The issue is not that it is mathematically impossible; it is that the number of candidates becomes too large. Search methods such as beam search reduce the number of prefixes retained, at the cost of no longer guaranteeing the global optimum.

## What to retain for the short answer

Use $N^T$ from 2.1, explain its growth with both $T$ and $N$, and conclude that exhaustive search is generally impractical for realistic vocabularies and sequence lengths. There is no need to commit to a detailed runtime model unless the cost of evaluating the language model is specified.
