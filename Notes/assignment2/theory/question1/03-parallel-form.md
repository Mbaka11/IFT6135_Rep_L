# Assignment 2 — Question 1.3: parallel form of linear attention

**Goal:** show that the matrix expression computes the same **unnormalized causal** outputs as the recurrent form from 1.2. We will expand one output row, then verify a small example.

- [Assignment statement and LaTeX answer](../../../../Assigments/IFT6135_A26__Student_/assignment2/en/assignment2_theory_en.tex)
- [Previous: 1.2 — Recurrent form](02-recurrent-form.md)
- [Slides: attention scores, PDF page 6](../../../../Slides/transformers.pdf#page=6)
- [Slides: causal decoder mask, PDF page 19](../../../../Slides/transformers.pdf#page=19)

**Status:** the concise answer for 1.3 is now in LaTeX.

## Quick reminders

- **Rows versus columns:** individual $\mathbf q_t,\mathbf k_i,\mathbf v_i,\mathbf o_t$ are $d\times1$ columns, but $\mathbf Q,\mathbf K,\mathbf V,\mathbf O$ stack their **transposes as rows**.
- The entry in row $t$, column $i$ of $\mathbf Q\mathbf K^\top$ is the scalar $\mathbf q_t^\top\mathbf k_i$: query $t$ compared with key $i$.
- The **Hadamard product** $\odot$ multiplies corresponding entries. Multiplying by the causal mask sets future-position scores to zero; it is not matrix multiplication.
- This question uses $\mathbf o_t=\mathbf S_t\mathbf q_t$ from 1.2, **without** softmax or the denominator from 1.1. The mask is applied to raw dot products, not to softmax logits.

## 1. Stack the projections as rows

The input sequence is represented by $\mathbf X\in\mathbb R^{L\times d}$, whose $t$-th row is $\mathbf x_t^\top$. Since $\mathbf q_t=\mathbf W_Q\mathbf x_t$,

$$
\mathbf q_t^\top=(\mathbf W_Q\mathbf x_t)^\top
=\mathbf x_t^\top\mathbf W_Q^\top.
$$

Thus the $t$-th row of $\mathbf X\mathbf W_Q^\top$ is $\mathbf q_t^\top$. Similarly,

$$
\mathbf Q=\mathbf X\mathbf W_Q^\top=\begin{bmatrix}\mathbf q_1^\top\\\vdots\\\mathbf q_L^\top\end{bmatrix},\quad
\mathbf K=\mathbf X\mathbf W_K^\top=\begin{bmatrix}\mathbf k_1^\top\\\vdots\\\mathbf k_L^\top\end{bmatrix},\quad
\mathbf V=\mathbf X\mathbf W_V^\top=\begin{bmatrix}\mathbf v_1^\top\\\vdots\\\mathbf v_L^\top\end{bmatrix}.
$$

All three have shape $L\times d$. The output matrix $\mathbf O$ has row $t$ equal to $\mathbf o_t^\top$.

## 2. A three-position example

Extend the 1.2 example with $\mathbf q_3=\mathbf k_3=[1,1]^\top$ and $\mathbf v_3=[0,1]^\top$. Stacking the rows gives

$$
\mathbf Q=\begin{bmatrix}1&1\\1&2\\1&1\end{bmatrix},\qquad
\mathbf K=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\qquad
\mathbf V=\begin{bmatrix}2&1\\1&3\\0&1\end{bmatrix}.
$$

First compute the score matrix, then apply the causal mask:

$$
\mathbf Q\mathbf K^\top=\begin{bmatrix}1&1&2\\1&2&3\\1&1&2\end{bmatrix},\qquad
\mathbf M=\begin{bmatrix}1&0&0\\1&1&0\\1&1&1\end{bmatrix},\qquad
\mathbf A=(\mathbf Q\mathbf K^\top)\odot\mathbf M
=\begin{bmatrix}1&0&0\\1&2&0\\1&1&2\end{bmatrix}.
$$

For example, the unmasked score $3$ at row 2, column 3 compares $\mathbf q_2$ with the **future** key $\mathbf k_3$; the mask changes it to zero. Finally,

$$
\mathbf O=\mathbf A\mathbf V
=\begin{bmatrix}2&1\\4&7\\3&6\end{bmatrix}.
$$

The second row is $\mathbf v_1^\top+2\mathbf v_2^\top=[4,7]$, matching the recurrent output $\mathbf o_2=[4,7]^\top$ from 1.2. It is a row here because $\mathbf O$ stacks transposed output vectors.

## 3. Why does the matrix expression work?

**Step 1 — score every query against every key.** From row-by-column multiplication,

$$
(\mathbf Q\mathbf K^\top)_{ti}=\mathbf q_t^\top\mathbf k_i.
$$

Both $\mathbf q_t^\top\mathbf k_i$ and $\mathbf k_i^\top\mathbf q_t$ are the same scalar dot product.

**Step 2 — remove future positions.** The assignment defines $M_{ti}=1$ for $i\le t$ and $M_{ti}=0$ for $i>t$. Therefore,

$$
A_{ti}=\bigl((\mathbf Q\mathbf K^\top)\odot\mathbf M\bigr)_{ti}
=\begin{cases}
\mathbf q_t^\top\mathbf k_i,&i\le t,\\
0,&i>t.
\end{cases}
$$

**Step 3 — multiply by the values.** Row $t$ of $\mathbf A\mathbf V$ is a weighted sum of the rows of $\mathbf V$:

$$
(\mathbf A\mathbf V)_{t,:}
=\sum_{i=1}^{L}A_{ti}\mathbf v_i^\top
=\sum_{i=1}^{t}(\mathbf q_t^\top\mathbf k_i)\mathbf v_i^\top.
$$

To compare it with the column-vector output, take the transpose:

$$
\begin{aligned}
\bigl((\mathbf A\mathbf V)_{t,:}\bigr)^\top
&=\sum_{i=1}^{t}\mathbf v_i(\mathbf k_i^\top\mathbf q_t)\\
&=\left(\sum_{i=1}^{t}\mathbf v_i\mathbf k_i^\top\right)\mathbf q_t\\
&=\mathbf S_t\mathbf q_t=\mathbf o_t.
\end{aligned}
$$

The equality holds for **every** row $t$, so the entire output matrix is

$$
\boxed{\mathbf O=(\mathbf Q\mathbf K^\top\odot\mathbf M)\mathbf V}.
$$

## 4. Dimension checks and a notation detail

| Quantity | Shape | Meaning |
| --- | --- | --- |
| $\mathbf Q,\mathbf K,\mathbf V$ | $L\times d$ | One transposed projection per row |
| $\mathbf Q\mathbf K^\top$, $\mathbf M$, $\mathbf A$ | $L\times L$ | Query–key scores, causal mask, masked scores |
| $\mathbf A\mathbf V=\mathbf O$ | $(L\times L)(L\times d)=L\times d$ | One transposed output per row |

The question labels $\mathbf O$ as $L\times d_v$ once, but earlier sets $\mathbf v_i,\mathbf o_i\in\mathbb R^d$ and $\mathbf W_V\in\mathbb R^{d\times d}$. Under these stated dimensions, **$d_v=d$**. More generally, if values had dimension $d_v$, $\mathbf V$ and $\mathbf O$ would be $L\times d_v$ and the same argument would still work; we do not change the assignment’s definitions.

The matrix form can evaluate many products together, but it constructs pairwise scores before masking. The detailed comparison of work and sequential steps is requested separately in **1.4**.

## What to retain for the short answer

Identify the $(t,i)$ score, show that the mask restricts the sum to $i\le t$, and transpose row $t$ to obtain $\mathbf S_t\mathbf q_t$. Then conclude that every row agrees, with the relevant matrix shapes.
