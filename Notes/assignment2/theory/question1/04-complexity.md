# Assignment 2 — Question 1.4: comparing attention complexity

**Goal:** compare the total computation and the number of sequential stages across the sequence for the three methods. We count the work **after** $\mathbf Q,\mathbf K,\mathbf V$ are available, as requested. This note covers 1.4; the LaTeX answer and table are now filled using the high-level stage convention explained below.

- [Assignment statement and LaTeX answer](../../../../Assigments/IFT6135_A26__Student_/assignment2/en/assignment2_theory_en.tex)
- [Previous: 1.2 — Recurrent form](02-recurrent-form.md)
- [Previous: 1.3 — Parallel form](03-parallel-form.md)
- [Course slide: attention and RNN operation counts, PDF page 23](../../../../Slides/transformers.pdf#page=23)

## Quick reminders

- $L$ is the sequence length; $d$ is the dimension of each query, key, value and output vector in this question.
- **Total work** counts all arithmetic operations, including work performed simultaneously. We use standard dense matrix multiplication, not specialized fast-matrix algorithms.
- **Sequential stages across positions** describe how many layers of sequence-dependent work must follow one another under an idealized, fully parallel implementation. This is **not** the same as elapsed GPU time or total work.
- $O(\cdot)$ describes asymptotic growth; it does not mean two methods with the same order take exactly the same time.
- The question says to ignore projection costs for $\mathbf Q,\mathbf K,\mathbf V$. Their standard dense computation would contribute $O(Ld^2)$ additional work; excluding it applies to **all three methods** in the comparison.

## 1. A small example: $L=4$, $d=2$

A causal output at position $t$ can use positions $1$ through $t$. Across the sequence, the number of allowed query–key pairs is

$$
1+2+3+4=10
=\frac{L(L+1)}{2}.
$$

A dense $4\times4$ score matrix instead has $4^2=16$ entries, including the six future-position entries that the causal mask removes. Both counts grow quadratically with $L$; a triangular implementation avoids some work but does not change the asymptotic order.

For the recurrent method, each of four positions updates a $2\times2$ state: that is $4\cdot2^2=16$ scalar multiplications for the outer products. Reading each state with a $2$-component query uses another $16$ multiplications. These are **illustrative multiplication counts**, not full FLOP counts: additions and other operations are excluded. The example helps explain the formulas below, not actual execution time.

## 2. Softmax attention: $O(L^2d)$ total work

The original causal attention uses query–key scores followed by a softmax and a weighted sum of values.

1. **Scores:** $\mathbf Q$ and $\mathbf K$ have shape $L\times d$, so $\mathbf Q\mathbf K^\top$ has $L^2$ entries. Each dot product involves $d$ components: $O(L^2d)$ work.
2. **Mask and softmax:** the $L\times L$ scores are masked and normalized across each row: $O(L^2)$ work in total.
3. **Output:** multiplying the $L\times L$ attention-weight matrix by $\mathbf V\in\mathbb R^{L\times d}$ costs $O(L^2d)$.

Therefore the total is $O(L^2d)$ for $d\ge1$. Causality restricts the allowed pairs to $L(L+1)/2$, which is still $O(L^2)$.

For this question, assume the **whole sequence and its projections are already available**. The rows can be processed concurrently: at the high level, score multiplication, masking/softmax, and value multiplication are a fixed number of stages, independent of $L$. This is an idealized **$O(1)$ sequence-level stage count**, not a claim that individual dot products, reductions or GPU execution take constant time.

## 3. Recurrent linear attention: $O(Ld^2)$ total work

The 1.2 algorithm starts from $\mathbf S_0=\mathbf0_{d\times d}$ and, at each $t$, performs

$$
\mathbf S_t=\mathbf S_{t-1}+\mathbf v_t\mathbf k_t^\top,
\qquad
\mathbf o_t=\mathbf S_t\mathbf q_t.
$$

- The outer product has $d^2$ entries: $O(d^2)$ work.
- Adding it to the previous state takes $O(d^2)$ work.
- Multiplying the $d\times d$ state by the query costs $O(d^2)$ work.

The loop has $L$ iterations, so its total work is **$O(Ld^2)$**, excluding the projection costs. In the **given sequential recurrence**, iteration $t$ needs $\mathbf S_{t-1}$ before forming $\mathbf S_t$: the dependency chain has **$O(L)$ sequence-level stages**. Updating before reading the state includes the current position.

## 4. Parallel linear attention: $O(L^2d)$ total work

The 1.3 formula is

$$
\mathbf O=(\mathbf Q\mathbf K^\top\odot\mathbf M)\mathbf V.
$$

Computing $\mathbf Q\mathbf K^\top$ costs $O(L^2d)$, applying the mask costs $O(L^2)$, and multiplying by $\mathbf V$ costs $O(L^2d)$. Thus the total is **$O(L^2d)$** with the dense expression. There is **no softmax** in this method.

With all projections available, the score product, mask, and value product are a fixed number of high-level stages: **$O(1)$ sequence-level stages** under the same idealized convention as softmax attention. The matrix method can use parallel hardware, but parallel execution does not eliminate the $O(L^2d)$ total arithmetic work.

## 5. Comparison and what “sequential steps” assumes

| Method | Total work, excluding projections | Sequence-level stages under the stated convention |
| --- | --- | --- |
| Causal softmax attention | $O(L^2d)$ | $O(1)$ with all positions available |
| Linear attention, given recurrent loop | $O(Ld^2)$ | $O(L)$ |
| Linear attention, given parallel matrix form | $O(L^2d)$ | $O(1)$ with all positions available |

**Important ambiguity:** the question does not specify a parallel-computation model or define “average number of sequential steps.” The table uses the common **layer/sequence-level** convention: a matrix multiplication or softmax is counted as a stage, without counting the internal arithmetic depth. It also assumes enough parallel resources and a full input sequence, not token-by-token generation. State this convention if using the table; the slide gives work orders, not a precise definition of sequential stages.

If instead we count **scalar-operation depth** with bounded fan-in, a balanced reduction over $d$ terms takes $O(\log d)$ depth, and softmax/weighted-value reductions over up to $L$ positions add $O(\log L)$ depth. Therefore the matrix approaches are **not literally constant-depth at the scalar-operation level**. Actual hardware has finite parallelism and communication costs, so runtime depends on implementation and device.

Also, $O(L)$ sequential stages describes the **specific loop supplied in 1.2**, not an unavoidable lower bound for every way to compute the same outputs. Once all key–value outer products are available, matrix addition is associative: a parallel prefix scan can compute the states in $O(\log L)$ sequence-depth with additional parallel resources. That is a different algorithm from the given recurrent loop.

For **autoregressive inference**, future tokens/projections are not known in advance. Generation itself must proceed token by token, so the full-sequence parallel-stage comparison does not directly describe end-to-end generation latency.

## 6. Compare growth without overclaiming

For fixed $d$, recurrent work grows linearly with $L$, whereas both dense matrix forms grow quadratically. Their work estimates cross in order of magnitude around $L\approx d$: when $L\gg d$, the recurrent count can be lower; when $d\gg L$, its $d^2$ factor can be larger. This is an **asymptotic comparison**, not a measured speed guarantee—matrix kernels, memory and hardware matter.

## What to retain for the short answer

Show where each work term comes from, complete the three rows, and explicitly state the full-sequence **sequence-level stage** convention. Keep the sequential loop distinct from alternative prefix-scan algorithms. Compare the dependence on $L$ while mentioning that $d$ and hardware also affect the practical trade-off.
