# Assignment 1 — Question 1.5: one gradient-descent update

**Scope:** Question 1, part (5): update $\mathbf b_1$, $W_1$, $b_2$, and $W_2$ once using the learning rate $\eta=0.5$.

- [Previous: 1.4 — Numerical gradients](04-numerical-gradients.md)
- [Next: 1.6 — Prediction after the update](06-prediction-after-update.md)
- [Assignment statement and answer template](../../../../Assigments/IFT6135_A26__Student_/assignment1/en/assignment1_theory_en.tex)

These are study notes. **The LaTeX answer for part (5) contains all four parameter updates.** We update the parameters here; the new forward pass belongs to part (6).

## 1. The update rule and its signs

For every parameter,

$$
\theta^{\mathrm{new}}
=\theta^{\mathrm{old}}-\eta\frac{\partial L}{\partial\theta},
\qquad \eta=\frac12.
$$

The derivative is evaluated at the **old** parameter values. For vectors or matrices, multiply every gradient entry by $1/2$ and subtract it from the corresponding parameter entry.

A useful sign check:

- A **negative** gradient makes the parameter increase, because subtracting a negative number adds a positive number.
- A **positive** gradient makes the parameter decrease.
- A zero gradient leaves the parameter unchanged in this vanilla update.

For example, $0-(1/2)(-1/2)=+1/4$, whereas $0-(1/2)(+1/2)=-1/4$.

### All updates use the same old forward pass

Compute all four gradients first, then use those saved gradients to update all four parameter groups. The order in which we write the calculations below does not change the result.

Do **not** update $W_2$ and then use $W_2^{\mathrm{new}}$ to recompute the hidden-layer gradients within this same step. That would mix two parameter states and would not be the requested single vanilla gradient-descent step.

## 2. Values we already know

The old parameters are

$$
\mathbf b_1=\begin{bmatrix}0\\0\\0\end{bmatrix},
\qquad
W_1=\begin{bmatrix}1&0\\1&1\\0&1\end{bmatrix},
\qquad b_2=0,
\qquad W_2=\begin{bmatrix}1\\-1\\1\end{bmatrix}.
$$

From [part (4)](04-numerical-gradients.md), the gradients are

$$
\frac{\partial L}{\partial\mathbf b_1}
=\begin{bmatrix}-\frac12\\\frac12\\-\frac12\end{bmatrix},
\qquad
\frac{\partial L}{\partial W_1}
=\begin{bmatrix}
-\frac12&-\frac12\\
\frac12&\frac12\\
-\frac12&-\frac12
\end{bmatrix},
$$

$$
\frac{\partial L}{\partial b_2}=-\frac12,
\qquad
\frac{\partial L}{\partial W_2}
=\begin{bmatrix}-\frac12\\-1\\-\frac12\end{bmatrix}.
$$

There is no additional batch-average factor or additional sigmoid derivative to introduce at this stage.

## 3. Update the hidden biases

Apply the rule to the whole vector:

$$
\begin{aligned}
\mathbf b_1^{\mathrm{new}}
&=\mathbf b_1-\frac12\frac{\partial L}{\partial\mathbf b_1}\\
&=\begin{bmatrix}0\\0\\0\end{bmatrix}
 -\frac12\begin{bmatrix}-\frac12\\\frac12\\-\frac12\end{bmatrix}\\
&=\boxed{\begin{bmatrix}\frac14\\-\frac14\\\frac14\end{bmatrix}}
=\begin{bmatrix}0.25\\-0.25\\0.25\end{bmatrix}.
\end{aligned}
$$

Entry by entry:

$$
b_{1,1}^{\mathrm{new}}=0-\frac12\left(-\frac12\right)=\frac14,
$$

$$
b_{1,2}^{\mathrm{new}}=0-\frac12\left(\frac12\right)=-\frac14,
$$

$$
b_{1,3}^{\mathrm{new}}=0-\frac12\left(-\frac12\right)=\frac14.
$$

The first and third biases increase; the second decreases. This follows the gradient signs, not the signs of the old biases, which were all zero.

## 4. Update the first-layer weights

$$
\begin{aligned}
W_1^{\mathrm{new}}
&=W_1-\frac12\frac{\partial L}{\partial W_1}\\
&=\begin{bmatrix}1&0\\1&1\\0&1\end{bmatrix}
 -\frac12\begin{bmatrix}
 -\frac12&-\frac12\\
 \frac12&\frac12\\
 -\frac12&-\frac12
 \end{bmatrix}\\
&=\begin{bmatrix}1&0\\1&1\\0&1\end{bmatrix}
 -\begin{bmatrix}
 -\frac14&-\frac14\\
 \frac14&\frac14\\
 -\frac14&-\frac14
 \end{bmatrix}\\
&=\boxed{\begin{bmatrix}
 \frac54&\frac14\\
 \frac34&\frac34\\
 \frac14&\frac54
 \end{bmatrix}}
=\begin{bmatrix}1.25&0.25\\0.75&0.75\\0.25&1.25\end{bmatrix}.
\end{aligned}
$$

Two useful entry-level checks:

$$
W_{1,12}^{\mathrm{new}}=0-\frac12\left(-\frac12\right)=\frac14,
\qquad
W_{1,21}^{\mathrm{new}}=1-\frac12\left(\frac12\right)=\frac34.
$$

A weight initialized to zero can become nonzero: its gradient is determined by the chain rule and the input, not by whether that weight itself is zero.

The first and third rows increase by $1/4$ in both columns. The second row decreases by $1/4$ in both columns. The matrix remains $3\times2$.

## 5. Update the output bias

$$
\begin{aligned}
b_2^{\mathrm{new}}
&=b_2-\frac12\frac{\partial L}{\partial b_2}\\
&=0-\frac12\left(-\frac12\right)\\
&=\boxed{\frac14=0.25}.
\end{aligned}
$$

The output bias increases. Holding the other quantities fixed, a larger output bias increases the logit and therefore increases the probability assigned to class 1.

## 6. Update the output weights

$$
\begin{aligned}
W_2^{\mathrm{new}}
&=W_2-\frac12\frac{\partial L}{\partial W_2}\\
&=\begin{bmatrix}1\\-1\\1\end{bmatrix}
 -\frac12\begin{bmatrix}-\frac12\\-1\\-\frac12\end{bmatrix}\\
&=\begin{bmatrix}1+\frac14\\-1+\frac12\\1+\frac14\end{bmatrix}\\
&=\boxed{\begin{bmatrix}\frac54\\-\frac12\\\frac54\end{bmatrix}}
=\begin{bmatrix}1.25\\-0.5\\1.25\end{bmatrix}.
\end{aligned}
$$

The middle calculation is worth checking explicitly:

$$
W_{2,2}^{\mathrm{new}}=-1-\frac12(-1)=-1+\frac12=-\frac12.
$$

All three output weights **increase numerically**. The middle weight becomes less negative; it does not become positive. Its change is $+1/2$, twice the $+1/4$ change of the other two output weights.

## 7. All updated parameters

| Parameter | New value | Shape |
| --- | --- | --- |
| $\mathbf b_1^{\mathrm{new}}$ | $[0.25,-0.25,0.25]^\top$ | $3\times1$ |
| $W_1^{\mathrm{new}}$ | $\begin{bmatrix}1.25&0.25\\0.75&0.75\\0.25&1.25\end{bmatrix}$ | $3\times2$ |
| $b_2^{\mathrm{new}}$ | $0.25$ | Scalar |
| $W_2^{\mathrm{new}}$ | $[1.25,-0.5,1.25]^\top$ | $3\times1$ |

All these decimals are exact representations of the fractions above; no rounding has been needed.

## 8. What changes, and what does not?

- The trainable parameters $W_1$, $\mathbf b_1$, $W_2$, and $b_2$ change.
- The input $\mathbf x=[1,1]^\top$ and target $y=1$ stay fixed.
- The hidden activations and predicted probability are **not** independent parameters to update by this rule. Recompute them from the new parameters in part (6).
- Do not apply a second gradient step: the question asks for one update only.

Subtracting the gradient gives a local descent direction, but a finite step is not automatically successful for every learning rate. Part (6) checks what actually happened for this network and this step.

## 9. What belongs in the LaTeX answer?

State $\eta=1/2$, then show each old parameter minus $\eta$ times its gradient and the resulting new value. The four displayed updates are the main content of the answer.

The detailed sign explanations remain here. **The LaTeX answer for part (5) is filled.**
