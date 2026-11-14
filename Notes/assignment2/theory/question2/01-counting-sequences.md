# Assignment 2 — Question 2.1: counting sequences

**Goal:** count sequences of total length $T+1$ that begin with a fixed beginning-of-sentence token. This note covers **2.1 only**.

- [Assignment statement](../../../../Assigments/IFT6135_A26__Student_/assignment2/en/assignment2_theory_en.tex)
- [Next: 2.2 — Exhaustive enumeration](02-exhaustive-enumeration.md)

**Status:** notes only; the LaTeX answer for 2.1 remains blank.

## Quick reminders

- $N$ is the vocabulary size, including the special token `<BOS>`.
- $w_0=\texttt{<BOS>}$ is **fixed**. The model then generates $T$ tokens $w_1,\ldots,w_T$.
- The **multiplication rule** says that if each of $T$ positions has $N$ independent choices, the number of possible ordered sequences is the product of those $T$ counts. “Independent choices” here refers to counting allowed combinations, **not** to the language model's conditional probabilities.

## 1. Small example

Let the vocabulary be $\{\texttt{<BOS>},a,b\}$, so $N=3$, and let $T=2$. The total sequence length is $3$: one fixed start token followed by two generated tokens.

For each choice of $w_1$, there are three choices of $w_2$:

| $w_1$ | Possible $w_2$ |
| --- | --- |
| `<BOS>` | `<BOS>`, $a$, $b$ |
| $a$ | `<BOS>`, $a$, $b$ |
| $b$ | `<BOS>`, $a$, $b$ |

There are $3\cdot3=9$ sequences. Their probabilities need not be equal; we are counting possibilities, not assigning probabilities.

## 2. General count

The start position has one choice, while each of the $T$ generated positions has $N$ choices. Therefore,

$$
\underbrace{1}_{w_0\text{ fixed}}
\cdot\underbrace{N\cdot N\cdots N}_{T\text{ generated positions}}
=\boxed{N^T}.
$$

This is an **exact count**, as requested, not Big-O notation. In particular, the exponent is $T$, not $T+1$: fixing $w_0$ does not introduce another factor of $N$.

## 3. Token restriction caveat

The question does not say that `<BOS>` is forbidden after position $0$, so the count above uses all $N$ vocabulary tokens at every generated position. If the course instead imposed a rule forbidding `<BOS>` in those positions, the count would be $(N-1)^T$. That is a **different assumption**, not a change made silently to the question.

Similarly, this counts all allowed token strings, even if a particular model assigns some of them zero probability. The number with **positive probability** can be smaller; the exact value would depend on the model. The stated count assumes no further token or termination restrictions.

## What to retain for the short answer

State that `<BOS>` is fixed and the other $T$ positions each have $N$ choices, giving $N^T$ sequences. Mention any restriction only if it is explicitly imposed.
