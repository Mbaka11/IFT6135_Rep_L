# Assignment 2 — Question 1.1: rewriting linear attention

**Goal:** start from the requested attention sum and show how to group its numerator and denominator into matrix form. This note covers **1.1 only**; the recurrence requested in 1.2 comes later.

- [Assignment statement and LaTeX answer location](../../../../Assigments/IFT6135_A26__Student_/assignment2/en/assignment2_theory_en.tex)
- [Slides: Q/K/V projections, PDF page 5](../../../../Slides/transformers.pdf#page=5)
- [Slides: causal decoder mask, PDF page 19](../../../../Slides/transformers.pdf#page=19)

**Status:** concise answers for 1.1–1.4 are now in LaTeX.

## Quick reminders

- At each position $i$, the **key** $\mathbf{k}_i$ is compared with the **query** $\mathbf{q}_t$ at the current position $t$ to produce a similarity score. The **value** $\mathbf{v}_i$ is the vector added to the output, weighted by that score.
- **Causal** means that at step $t$, only positions $1,\ldots,t$ are used: no future position contributes.
- All vectors in this question are **columns** of dimension $d$. The product $\mathbf{k}_i^\top\mathbf{q}_t$ is a **scalar**; the product $\mathbf{v}_i\mathbf{k}_i^\top$ is a $d\times d$ **matrix**.
- In this particular question, linear attention replaces $\exp(\mathbf{k}_i^\top\mathbf{q}_t)$ with $\mathbf{k}_i^\top\mathbf{q}_t$. The scaling factor sometimes present in other formulations does not appear in the assignment equation, so we do not add it here.

## 1. A small example before the general formula

Let $t=2$ and $d=2$, with

$$
\mathbf{q}_2=\begin{bmatrix}1\\2\end{bmatrix},\quad
\mathbf{k}_1=\begin{bmatrix}1\\0\end{bmatrix},\quad
\mathbf{k}_2=\begin{bmatrix}0\\1\end{bmatrix},\quad
\mathbf{v}_1=\begin{bmatrix}2\\1\end{bmatrix},\quad
\mathbf{v}_2=\begin{bmatrix}1\\3\end{bmatrix}.
$$

The two scores are $\mathbf{k}_1^\top\mathbf{q}_2=1$ and $\mathbf{k}_2^\top\mathbf{q}_2=2$. Their sum is $3$. After replacing the exponential, the output is

$$
\mathbf{o}_2=\frac{1}{3}\mathbf{v}_1+\frac{2}{3}\mathbf{v}_2
=\frac{1}{3}\begin{bmatrix}2\\1\end{bmatrix}
 +\frac{2}{3}\begin{bmatrix}1\\3\end{bmatrix}
=\begin{bmatrix}4/3\\7/3\end{bmatrix}.
$$

**Idea:** the denominators are identical, so we can first add the weighted values and then divide by $3$. The following steps show how to obtain the same result from an accumulated matrix and vector.

## 2. Make the requested replacement

At step $t$, the assignment uses the same query $\mathbf{q}_t$ to compare keys $\mathbf{k}_1,\ldots,\mathbf{k}_t$. Replacing each exponential with the corresponding dot product gives

$$
\mathbf{o}_t
=\sum_{i=1}^{t}
\frac{\mathbf{k}_i^\top\mathbf{q}_t}
{\displaystyle\sum_{j=1}^{t}\mathbf{k}_j^\top\mathbf{q}_t}\,\mathbf{v}_i.
$$

Both $i$ and $j$ range over the past and current positions. They use different letters because $j$ is only used to calculate the **common denominator sum**, while $i$ runs over the terms in the numerator. The denominator therefore does not depend on $i$.

Assuming the denominator is nonzero, we can take it outside the sum:

$$
\mathbf{o}_t
=\frac{\displaystyle\sum_{i=1}^{t}
(\mathbf{k}_i^\top\mathbf{q}_t)\mathbf{v}_i}
{\displaystyle\sum_{j=1}^{t}\mathbf{k}_j^\top\mathbf{q}_t}.
$$

## 3. Group the numerator: where does $\mathbf{S}_t$ come from?

The score $\mathbf{k}_i^\top\mathbf{q}_t$ is a number. We can therefore move this scalar before or after the vector $\mathbf{v}_i$, then use associativity:

$$
(\mathbf{k}_i^\top\mathbf{q}_t)\mathbf{v}_i
=\mathbf{v}_i(\mathbf{k}_i^\top\mathbf{q}_t)
=(\mathbf{v}_i\mathbf{k}_i^\top)\mathbf{q}_t.
$$

This does **not** change the order of the matrices: we only regrouped the product, keeping $\mathbf{v}_i$, then $\mathbf{k}_i^\top$, then $\mathbf{q}_t$. The query is the same for every $i$, so distributivity gives

$$
\sum_{i=1}^{t}(\mathbf{k}_i^\top\mathbf{q}_t)\mathbf{v}_i
=\sum_{i=1}^{t}(\mathbf{v}_i\mathbf{k}_i^\top)\mathbf{q}_t
=\left(\sum_{i=1}^{t}\mathbf{v}_i\mathbf{k}_i^\top\right)\mathbf{q}_t.
$$

The assignment calls this sum of matrices

$$
\mathbf{S}_t=\sum_{i=1}^{t}\mathbf{v}_i\mathbf{k}_i^\top,
\qquad \text{so the numerator is }\mathbf{S}_t\mathbf{q}_t.
$$

**Dimensions:** $(d\times1)(1\times d)=d\times d$ for each outer product $\mathbf{v}_i\mathbf{k}_i^\top$, so $\mathbf{S}_t$ is $d\times d$. Then $(d\times d)(d\times1)=d\times1$: the numerator is a vector with the same dimension as the output.

## 4. Group the denominator: where does $\mathbf{z}_t$ come from?

The query $\mathbf{q}_t$ is also common to every term in the denominator. By distributivity,

$$
\sum_{j=1}^{t}\mathbf{k}_j^\top\mathbf{q}_t
=\left(\sum_{j=1}^{t}\mathbf{k}_j^\top\right)\mathbf{q}_t
=\left(\sum_{j=1}^{t}\mathbf{k}_j\right)^\top\mathbf{q}_t.
$$

Define $\mathbf{z}_t=\sum_{j=1}^{t}\mathbf{k}_j\in\mathbb{R}^{d}$. Its product with the query is a scalar: $(1\times d)(d\times1)=1\times1$. Combining the two results,

$$
\boxed{\mathbf{o}_t=\frac{\mathbf{S}_t\mathbf{q}_t}{\mathbf{z}_t^\top\mathbf{q}_t}},
\qquad \text{if }\mathbf{z}_t^\top\mathbf{q}_t\ne0.
$$

**Checking the example:**

$$
\mathbf{S}_2
=\begin{bmatrix}2&0\\1&0\end{bmatrix}
+\begin{bmatrix}0&1\\0&3\end{bmatrix}
=\begin{bmatrix}2&1\\1&3\end{bmatrix},
\qquad
\mathbf{z}_2=\begin{bmatrix}1\\1\end{bmatrix}.
$$

Thus $\mathbf{S}_2\mathbf{q}_2=[4,7]^\top$ and $\mathbf{z}_2^\top\mathbf{q}_2=3$, giving the same $\mathbf{o}_2=[4/3,7/3]^\top$ as before.

## 5. Caution: the scores are not always probabilities

In the original softmax, the exponentials are positive and their sum cannot be zero. Here, a dot product can be **negative or zero**: normalized weights can therefore be negative, and the denominator can be zero. For example, if $\mathbf{q}_t=\mathbf{0}$, all scores and their sum are zero, so the fraction is undefined. The result in 1.1 holds when its denominator is nonzero; with $\phi=\mathrm{id}$, this replacement does not generally produce a probability distribution.

For **the rest** of the exercise, the assignment chooses to drop the denominator and use $\mathbf{o}_t=\mathbf{S}_t\mathbf{q}_t$. This is a new convention for 1.2–1.4, **not** a general equality with the normalized output in 1.1.

## What to retain for the short answer

1. Replace the exponentials and take the common denominator outside the sum.
2. Rewrite each numerator term as $(\mathbf{v}_i\mathbf{k}_i^\top)\mathbf{q}_t$, then factor out the query.
3. Factor out the same query in the denominator and identify $\mathbf{S}_t$ and $\mathbf{z}_t$. State that the denominator must be nonzero.
