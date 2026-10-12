# Assignment 1 — Question 1.6: prediction after the update

**Scope:** Question 1, part (6): run a new forward pass on the same sample using the parameters from part (5), then check whether the prediction moved toward the correct label.

- [Previous: 1.5 — One gradient-descent update](05-gradient-descent-update.md)
- [Initial forward pass: 1.1](01-forward-pass.md)
- [Assignment statement and answer template](../../../Assigments/IFT6135_A26__Student_/assignment1/en/assignment1_theory_en.tex)

These are study notes. **The LaTeX answer for part (6) contains the new forward pass and the improvement conclusion.** We do not take another gradient step here.

## 1. What stays fixed and what changes?

Use the same input and target:

$$
\mathbf x=\begin{bmatrix}1\\1\end{bmatrix},
\qquad y=1.
$$

Use **all four updated parameter groups**, not a mixture of old and new parameters:

$$
W_1^{\mathrm{new}}=
\begin{bmatrix}\frac54&\frac14\\\frac34&\frac34\\\frac14&\frac54\end{bmatrix},
\qquad
\mathbf b_1^{\mathrm{new}}=
\begin{bmatrix}\frac14\\-\frac14\\\frac14\end{bmatrix},
$$

$$
W_2^{\mathrm{new}}=
\begin{bmatrix}\frac54\\-\frac12\\\frac54\end{bmatrix},
\qquad b_2^{\mathrm{new}}=\frac14.
$$

The forward-pass equations themselves have not changed. The superscript “new” simply marks quantities computed after the single parameter update.

## 2. New hidden pre-activations

Start with the first affine layer:

$$
\mathbf z^{(1),\mathrm{new}}
=W_1^{\mathrm{new}}\mathbf x+\mathbf b_1^{\mathrm{new}}.
$$

Substitute the numbers:

$$
\begin{aligned}
\mathbf z^{(1),\mathrm{new}}
&=\begin{bmatrix}
\frac54&\frac14\\
\frac34&\frac34\\
\frac14&\frac54
\end{bmatrix}
\begin{bmatrix}1\\1\end{bmatrix}
+\begin{bmatrix}\frac14\\-\frac14\\\frac14\end{bmatrix}\\
&=\begin{bmatrix}\frac32\\\frac32\\\frac32\end{bmatrix}
+\begin{bmatrix}\frac14\\-\frac14\\\frac14\end{bmatrix}\\
&=\boxed{\begin{bmatrix}\frac74\\\frac54\\\frac74\end{bmatrix}}
=\begin{bmatrix}1.75\\1.25\\1.75\end{bmatrix}.
\end{aligned}
$$

An entry-level check for the middle neuron is

$$
z_2^{(1),\mathrm{new}}
=\frac34\cdot1+\frac34\cdot1-\frac14
=\frac54=1.25.
$$

Its bias is negative, but its total pre-activation is still positive.

## 3. New hidden activations

Apply ReLU component-wise:

$$
\mathbf h^{(1),\mathrm{new}}
=\operatorname{ReLU}(\mathbf z^{(1),\mathrm{new}}).
$$

All three entries are positive, so

$$
\boxed{
\mathbf h^{(1),\mathrm{new}}
=\begin{bmatrix}\frac74\\\frac54\\\frac74\end{bmatrix}
=\begin{bmatrix}1.75\\1.25\\1.75\end{bmatrix}.
}
$$

Compared with the original $[1,2,1]^\top$, the first and third activations increased, while the middle activation decreased. The forward pass uses these activation **values**, not the all-ones derivative mask used in backpropagation.

## 4. New output logit

The output is a dot product plus the updated output bias:

$$
z^{(2),\mathrm{new}}
=(W_2^{\mathrm{new}})^\top\mathbf h^{(1),\mathrm{new}}
+b_2^{\mathrm{new}}.
$$

Expand every contribution:

$$
\begin{aligned}
z^{(2),\mathrm{new}}
&=\frac54\cdot\frac74
 -\frac12\cdot\frac54
 +\frac54\cdot\frac74
 +\frac14\\
&=\frac{35}{16}-\frac{10}{16}
 +\frac{35}{16}+\frac{4}{16}\\
&=\boxed{\frac{64}{16}=4}.
\end{aligned}
$$

In decimals, the same calculation is

$$
1.25(1.75)-0.5(1.25)+1.25(1.75)+0.25
=2.1875-0.625+2.1875+0.25=4.
$$

### Where did the increase from 0 to 4 come from?

| Contribution to the output logit | Before the update | After the update |
| -------------------------------- | ----------------- | ---------------- |
| First hidden neuron              | $1$               | $35/16=2.1875$   |
| Second hidden neuron             | $-2$              | $-5/8=-0.625$    |
| Third hidden neuron              | $1$               | $35/16=2.1875$   |
| Output bias                      | $0$               | $1/4=0.25$       |
| **Total**                        | **$0$**           | **$4$**          |

The positive contributions grew, the negative contribution became less negative, and the output bias increased. All of these changes push the logit upward in this example.

## 5. New predicted probability: use the assignment's approximation

First write the standard sigmoid expression:

$$
\hat y^{\mathrm{new}}
=\sigma(4)=\frac{1}{1+e^{-4}}.
$$

The assignment tells us to approximate $e\approx3$. Apply that convention **when evaluating the exponential numerically**:

$$
e^{-4}=\frac{1}{e^4}\approx\frac{1}{3^4}=\frac{1}{81}.
$$

Therefore,

$$
\boxed{
\hat y^{\mathrm{new}}
\approx\frac{1}{1+1/81}
=\frac{81}{82}
\approx0.9878.
}
$$

Use $\approx$, not $=$, between the standard sigmoid value and $81/82$: the fraction is exact for the prescribed approximation, not for the true mathematical value of $e$.

### Why might a calculator show about 0.9820 instead?

A calculator using the usual $e\approx2.71828$ gives

$$
\sigma(4)\approx0.98201379.
$$

That is the standard sigmoid value. For this assignment's numerical answer, follow the stated $e\approx3$ convention and report $81/82\approx0.9878$.

This convention does not redefine the sigmoid for the symbolic differentiation in parts (2)–(3). It is a numerical approximation used after those derivatives have been derived.

## 6. Did the update move the prediction in the right direction?

**Yes.** The target is $y=1$. The probability assigned to class 1 increased from

$$
\hat y^{\mathrm{old}}=0.5
\qquad\text{to}\qquad
\hat y^{\mathrm{new}}\approx0.9878
$$

under the assignment's approximation. The prediction is now much closer to 1.

This conclusion does not depend on the approximation: the logit increased from 0 to 4, and the standard sigmoid is increasing, so $\sigma(4)>\sigma(0)=1/2$ as well.

The sigmoid output is a **probability**. If one additionally uses a 0.5 decision threshold, the new hard class prediction is 1. The numerical forward-pass result to report is the probability, not just the hard label.

A finite sigmoid input does not produce probability exactly 1; the update does not need to reach 1 to move in the correct direction.

### Optional check: the loss also decreased

Because $y=1$, binary cross-entropy is $L=-\ln\hat y$. Initially,

$$
L^{\mathrm{old}}=-\ln\frac12=\ln2\approx0.6931.
$$

With the standard sigmoid after the update,

$$
L^{\mathrm{new}}=\ln(1+e^{-4})<\ln2,
$$

since $e^{-4}<1$ and the logarithm is increasing. Under the assignment's numerical approximation,

$$
L^{\mathrm{new}}\approx-\ln\frac{81}{82}
=\ln\frac{82}{81}\approx0.01227.
$$

The logarithm remains the **natural logarithm**; approximating $e$ in the exponential is not a reason to switch the loss to a base-3 logarithm.

This is an extra verification, not a separately requested loss calculation.

## 7. New forward-pass summary

| Quantity                       | Value after one update                                 |
| ------------------------------ | ------------------------------------------------------ |
| $\mathbf z^{(1),\mathrm{new}}$ | $[7/4,5/4,7/4]^\top=[1.75,1.25,1.75]^\top$             |
| $\mathbf h^{(1),\mathrm{new}}$ | $[7/4,5/4,7/4]^\top=[1.75,1.25,1.75]^\top$             |
| $z^{(2),\mathrm{new}}$         | $4$                                                    |
| $\hat y^{\mathrm{new}}$        | $\sigma(4)\approx81/82\approx0.9878$ using $e\approx3$ |

## 8. Common mistakes and submission guidance

- **Reusing the old hidden activations** after changing the first-layer parameters.
- **Mixing old and new parameters**, or forgetting an updated bias.
- **Applying the ReLU derivative** instead of ReLU itself in the new forward pass.
- **Computing $3^{-4}$ as $1/12$.** It is $1/3^4=1/81$.
- **Treating the logit 4 as the predicted probability.** Apply sigmoid afterward.
- **Taking another gradient step.** Part (6) asks for a new prediction, not another update.

For a concise LaTeX answer, show the new hidden vector, the logit calculation giving 4, the sigmoid evaluation with $e\approx3$, and one sentence explaining that the probability increased toward $y=1$.

**The LaTeX answer for part (6) is filled.**
