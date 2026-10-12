# Assignment 1 — Question 1.4: numerical gradients

**Scope:** Question 1, part (4): evaluate the four parameter gradients using the initial parameters and the sample $\mathbf{x}=[1,1]^\top$, $y=1$. This is not the separate main Question 4.

- [Previous: 1.3 — Analytical gradients](03-analytical-gradients.md)
- [Next: 1.5 — One gradient-descent update](05-gradient-descent-update.md)
- [Assignment statement and answer template](../../../Assigments/IFT6135_A26__Student_/assignment1/en/assignment1_theory_en.tex)

These are study notes. **The LaTeX answer for part (4) contains the four numerical parameter gradients and their substitutions.** We calculate gradients here; we do not update the parameters until part (5).

## 1. What changes from part (3)?

Part (3) derived formulas using symbols. Part (4) substitutes the numbers into those same formulas. There is no new differentiation rule to learn.

Use the **initial** forward pass from [part (1)](01-forward-pass.md):

$$
\mathbf{z}^{(1)}=\mathbf{h}^{(1)}=
\begin{bmatrix}1\\2\\1\end{bmatrix},
\qquad z^{(2)}=0,
\qquad \hat y=\frac12.
$$

The initial output weights and input are

$$
W_2=\begin{bmatrix}1\\-1\\1\end{bmatrix},
\qquad \mathbf{x}=\begin{bmatrix}1\\1\end{bmatrix},
\qquad y=1.
$$

From [part (2)](02-cross-entropy-derivative.md),

$$
\frac{\partial L}{\partial z^{(2)}}=\hat y-y
=\frac12-1=-\frac12.
$$

All three hidden pre-activations are positive, so

$$
\mathbf r=\operatorname{ReLU}'(\mathbf z^{(1)})
=\begin{bmatrix}1\\1\\1\end{bmatrix}.
$$

**The mask is $[1,1,1]^\top$, not $[1,2,1]^\top$.** ReLU's derivative is 1 on its positive branch, regardless of the activation's magnitude. The convention at zero is not needed at this initialization.

## 2. Output bias: $\partial L/\partial b_2$

The scalar chain is

$$
\frac{\partial L}{\partial b_2}
=\frac{\partial L}{\partial z^{(2)}}
 \frac{\partial z^{(2)}}{\partial b_2}.
$$

Because $z^{(2)}=W_2^\top\mathbf h^{(1)}+b_2$, its derivative with respect to $b_2$ is 1. Therefore,

$$
\boxed{\frac{\partial L}{\partial b_2}
=\left(-\frac12\right)\cdot1=-\frac12.}
$$

The result is a scalar, just like $b_2$.

## 3. Output weights: $\partial L/\partial W_2$

For each component,

$$
\frac{\partial L}{\partial W_{2,i}}
=\frac{\partial L}{\partial z^{(2)}}
 \frac{\partial z^{(2)}}{\partial W_{2,i}}
=\left(-\frac12\right)h_i^{(1)}.
$$

Substitute the three hidden activations:

$$
\frac{\partial L}{\partial W_{2,1}}=-\frac12\cdot1=-\frac12,
\qquad
\frac{\partial L}{\partial W_{2,2}}=-\frac12\cdot2=-1,
\qquad
\frac{\partial L}{\partial W_{2,3}}=-\frac12\cdot1=-\frac12.
$$

Stacking these scalar derivatives into a column gives

$$
\boxed{
\frac{\partial L}{\partial W_2}
=-\frac12\begin{bmatrix}1\\2\\1\end{bmatrix}
=\begin{bmatrix}-\frac12\\-1\\-\frac12\end{bmatrix}.
}
$$

**Why is the middle gradient twice as large?** Its hidden activation is 2 instead of 1. Changing that output weight by a small amount changes the logit twice as much.

Notice that $W_{2,2}=-1$ does **not** appear in its own gradient formula: $\partial z^{(2)}/\partial W_{2,2}=h_2^{(1)}$, not $W_{2,2}h_2^{(1)}$.

## 4. Hidden biases: $\partial L/\partial\mathbf b_1$

Here we follow the explicitly expanded chain from part (3):

$$
\begin{aligned}
\frac{\partial L}{\partial b_{1,i}}
&=\frac{\partial L}{\partial z^{(2)}}
  \frac{\partial z^{(2)}}{\partial h_i^{(1)}}
  \frac{\partial h_i^{(1)}}{\partial z_i^{(1)}}
  \frac{\partial z_i^{(1)}}{\partial b_{1,i}}\\
&=\left(-\frac12\right)W_{2,i}r_i\cdot1.
\end{aligned}
$$

Each local derivative is scalar. Substituting neuron by neuron:

| Neuron $i$ | Output weight $W_{2,i}$ | ReLU derivative $r_i$ | $\partial L/\partial b_{1,i}$ |
| --- | --- | --- | --- |
| 1 | $1$ | $1$ | $(-1/2)(1)(1)(1)=-1/2$ |
| 2 | $-1$ | $1$ | $(-1/2)(-1)(1)(1)=+1/2$ |
| 3 | $1$ | $1$ | $(-1/2)(1)(1)(1)=-1/2$ |

Thus,

$$
\boxed{
\frac{\partial L}{\partial\mathbf b_1}
=\left(-\frac12 W_2\right)\odot\mathbf r
=\begin{bmatrix}-\frac12\\\frac12\\-\frac12\end{bmatrix}.
}
$$

### Why is the middle sign positive?

The output error $\partial L/\partial z^{(2)}$ is negative, but the middle output weight is also negative. Their product is positive.

Increasing the second hidden bias increases the second hidden activation. Because its output weight is negative, this lowers $z^{(2)}$, lowers the probability assigned to the correct label $y=1$, and increases the loss. The positive gradient reflects that local effect.

For comparison, the hidden-activation gradient is

$$
\frac{\partial L}{\partial\mathbf h^{(1)}}
=-\frac12 W_2
=\begin{bmatrix}-\frac12\\\frac12\\-\frac12\end{bmatrix}.
$$

It equals the bias gradient **in this sample** because all entries of the ReLU mask are 1. Part (4) does not request this intermediate gradient separately, and the equality would not generally hold with inactive hidden neurons.

## 5. First-layer weights: $\partial L/\partial W_1$

The expanded chain for an entry in row $i$, column $j$ is

$$
\begin{aligned}
\frac{\partial L}{\partial W_{1,ij}}
&=\frac{\partial L}{\partial z^{(2)}}
  \frac{\partial z^{(2)}}{\partial h_i^{(1)}}
  \frac{\partial h_i^{(1)}}{\partial z_i^{(1)}}
  \frac{\partial z_i^{(1)}}{\partial W_{1,ij}}\\
&=\left(-\frac12\right)W_{2,i}r_i x_j.
\end{aligned}
$$

The first three factors are exactly the hidden-bias gradient for neuron $i$; the last factor is $x_j$ instead of 1:

$$
\frac{\partial L}{\partial W_{1,ij}}
=\frac{\partial L}{\partial b_{1,i}}x_j.
$$

Since $x_1=x_2=1$, both columns have the same values. In matrix form,

$$
\begin{aligned}
\frac{\partial L}{\partial W_1}
&=\frac{\partial L}{\partial\mathbf b_1}\mathbf x^\top\\
&=\begin{bmatrix}-\frac12\\\frac12\\-\frac12\end{bmatrix}
  \begin{bmatrix}1&1\end{bmatrix}\\
&=\boxed{\begin{bmatrix}
-\frac12&-\frac12\\
\frac12&\frac12\\
-\frac12&-\frac12
\end{bmatrix}}.
\end{aligned}
$$

This is an **outer product**: $(3\times1)(1\times2)=3\times2$, matching the shape of $W_1$.

As an entry-level check, $\partial L/\partial W_{1,21}=(-1/2)(-1)(1)(1)=+1/2$. The row index selects hidden neuron 2; the column index selects input 1.

## 6. All four requested results

| Parameter | Numerical gradient | Shape |
| --- | --- | --- |
| $b_2$ | $-1/2$ | Scalar |
| $W_2$ | $[-1/2,-1,-1/2]^\top$ | $3\times1$ |
| $\mathbf b_1$ | $[-1/2,1/2,-1/2]^\top$ | $3\times1$ |
| $W_1$ | $\begin{bmatrix}-1/2&-1/2\\1/2&1/2\\-1/2&-1/2\end{bmatrix}$ | $3\times2$ |

These are exact fractions. The approximation $e\approx3$ introduces no numerical difference here because the initial logit is zero and $\hat y=1/2$ exactly.

## 7. Common mistakes and self-checks

- **Using $W_2$ instead of $\mathbf h^{(1)}$ for the output-weight gradient.** The local derivative with respect to a weight is its input activation.
- **Dropping the negative middle output weight** when propagating backward to the hidden layer.
- **Using the activation values as the ReLU mask.** The mask is $[1,1,1]^\top$.
- **Multiplying by the sigmoid derivative again.** It is already included in $\partial L/\partial z^{(2)}=-1/2$.
- **Multiplying these gradients by $\eta$ now.** The learning rate belongs to the update in part (5), not the derivative itself.
- **Recomputing some gradients after changing a parameter.** All gradients above belong to the same initial forward pass.

Check that you can explain why the output-weight gradient's middle entry is $-1$, while the hidden-bias gradient's middle entry is $+1/2$.

## 8. What belongs in the LaTeX answer?

For part (4), show the four numerical gradients with their substitutions. It is enough to recall $\hat y-y=-1/2$ and the all-ones ReLU mask; the full symbolic chain rules are already in part (3).

The explanations and sign checks above remain in these notes. **The LaTeX answer for part (4) is filled.**
