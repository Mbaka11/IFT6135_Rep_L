# Assignment 2 — Question 1.2: recurrent form of linear attention

**Goal:** compute the outputs one position at a time by updating a matrix state, rather than rebuilding its full sum at every step. This note covers **1.2 only**; the detailed complexity comparison belongs to 1.4.

- [Assignment statement and LaTeX answer](../../../../Assigments/IFT6135_A26__Student_/assignment2/en/assignment2_theory_en.tex)
- [Previous: 1.1 — Rewriting linear attention](01-attention-lineaire.md)
- [RNN slides: recurrent hidden state, PDF page 10](../../../../Slides/05_RNN.pdf#page=10)

**Status:** the concise 1.2 answer and the supplied algorithm blanks are now filled in LaTeX.

## Quick reminders

- A **state** is information carried from one step to the next. Here it is a matrix, not necessarily a vector as in a familiar RNN.
- At position $t$, the query $\mathbf q_t$, key $\mathbf k_t$, and value $\mathbf v_t$ are column vectors in $\mathbb R^d$, computed from the current input $\mathbf x_t$.
- The assignment now uses **unnormalized** linear attention: $\mathbf o_t=\mathbf S_t\mathbf q_t$. Do not include the denominator from 1.1 in this part.
- An **outer product** $\mathbf v_t\mathbf k_t^\top$ has shape $(d\times1)(1\times d)=d\times d$. Each entry in its row $r$, column $c$ is $(v_t)_r(k_t)_c$.

## 1. A small two-step example

Reuse the keys and values from 1.1, and choose a first query:

$$
\mathbf k_1=\begin{bmatrix}1\\0\end{bmatrix},\quad
\mathbf k_2=\begin{bmatrix}0\\1\end{bmatrix},\quad
\mathbf v_1=\begin{bmatrix}2\\1\end{bmatrix},\quad
\mathbf v_2=\begin{bmatrix}1\\3\end{bmatrix},\quad
\mathbf q_1=\begin{bmatrix}1\\1\end{bmatrix},\quad
\mathbf q_2=\begin{bmatrix}1\\2\end{bmatrix}.
$$

Start with the $2\times2$ zero matrix. At $t=1$, add the first outer product and use the updated state:

$$
\mathbf S_0=\begin{bmatrix}0&0\\0&0\end{bmatrix},\qquad
\mathbf S_1=\mathbf S_0+\mathbf v_1\mathbf k_1^\top
=\begin{bmatrix}2&0\\1&0\end{bmatrix},\qquad
\mathbf o_1=\mathbf S_1\mathbf q_1=\begin{bmatrix}2\\1\end{bmatrix}.
$$

At $t=2$, keep $\mathbf S_1$ and add only the second outer product:

$$
\mathbf S_2=\underbrace{\begin{bmatrix}2&0\\1&0\end{bmatrix}}_{\mathbf S_1}
+\underbrace{\begin{bmatrix}0&1\\0&3\end{bmatrix}}_{\mathbf v_2\mathbf k_2^\top}
=\begin{bmatrix}2&1\\1&3\end{bmatrix},\qquad
\mathbf o_2=\mathbf S_2\mathbf q_2=\begin{bmatrix}4\\7\end{bmatrix}.
$$

The output is $[4,7]^\top$, **not** $[4/3,7/3]^\top$ as in 1.1: this part drops the normalization denominator.

## 2. Derive the state recurrence

The assignment defines $\mathbf S_t$ as the sum of all outer products up to and including position $t$. Separate its last term:

$$
\begin{aligned}
\mathbf S_t
&=\sum_{i=1}^{t}\mathbf v_i\mathbf k_i^\top\\
&=\underbrace{\sum_{i=1}^{t-1}\mathbf v_i\mathbf k_i^\top}_{\mathbf S_{t-1}}
  +\mathbf v_t\mathbf k_t^\top\\
&=\mathbf S_{t-1}+\mathbf v_t\mathbf k_t^\top.
\end{aligned}
$$

Before any position has been processed, the sum has no terms. Its value is the **zero matrix**, so $\mathbf S_0=\mathbf 0_{d\times d}$. It is not the identity matrix: the state should contain no past contribution at initialization.

The recurrence reuses the previous sum and adds the current key–value contribution. The current **query is not stored** in the state: it is used afterward to read the updated state, $\mathbf o_t=\mathbf S_t\mathbf q_t$.

## 3. Algorithm and order of operations

For a sequence $\mathbf x_1,\ldots,\mathbf x_L$ and the assignment’s projection matrices $\mathbf W_Q,\mathbf W_K,\mathbf W_V\in\mathbb R^{d\times d}$:

1. Initialize $\mathbf S\leftarrow\mathbf 0_{d\times d}$.
2. For $t=1,\ldots,L$:
   - Compute $\mathbf q_t=\mathbf W_Q\mathbf x_t$, $\mathbf k_t=\mathbf W_K\mathbf x_t$, and $\mathbf v_t=\mathbf W_V\mathbf x_t$.
   - Update $\mathbf S\leftarrow\mathbf S+\mathbf v_t\mathbf k_t^\top$.
   - Compute $\mathbf o_t\leftarrow\mathbf S\mathbf q_t$ and save it.
3. Return $\mathbf o_1,\ldots,\mathbf o_L$.

**Why update before computing the output?** Causal attention includes the **current** position $i=t$. Reading $\mathbf S_{t-1}$ first would omit $\mathbf v_t\mathbf k_t^\top$ and produce a strictly past-only result instead.

## 4. Dimension and interpretation checks

| Quantity | Shape | Role |
| --- | --- | --- |
| $\mathbf x_t,\mathbf q_t,\mathbf k_t,\mathbf v_t$ | $d\times1$ | Current input and projections |
| $\mathbf v_t\mathbf k_t^\top$, $\mathbf S_t$ | $d\times d$ | Current contribution and accumulated state |
| $\mathbf S_t\mathbf q_t=\mathbf o_t$ | $d\times1$ | Output at position $t$ |

The algorithm processes $L$ positions in one loop and performs a fixed set of operations at each position for fixed $d$, so its work grows linearly with $L$. **This is an RNN-like recurrence**, not a claim that it is a standard nonlinear RNN or that the state recovers each previous value separately. The detailed dependence on $d$ and comparison with other methods are for 1.4.

## What to retain for the short answer

Show how the last term is separated from the sum, state $\mathbf S_0=\mathbf0$, and give the algorithm in the correct order: project, **update**, then compute the output. Mention that the single pass is linear in $L$ for fixed $d$.
