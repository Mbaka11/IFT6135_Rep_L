# Assignment 2 — Question 4.2: squared-error level sets

**Goal:** prove when the unregularized squared-error loss in the two parameters $(\beta_0,\beta_1)$ has elliptical level sets, and identify the cases in which the claim needs qualification.

- [Assignment statement](../../../../Assigments/IFT6135_A26__Student_/assignment2/en/assignment2_theory_en.tex)
- [Previous: 4.1 — Weight decay and $L_2$](01-weight-decay-and-l2.md)
- [Next: 4.3 — Sparsity](03-sparsity-and-feature-selection.md)

**Status:** notes only; the Question 4 LaTeX answer remains blank.

## 1. What is a level set?

For a function $L(\beta_0,\beta_1)$, a level set at value $c$ is the set of all parameter pairs with the **same** loss:

$$
\{(\beta_0,\beta_1):L(\beta_0,\beta_1)=c\}.
$$

Here there is one intercept and one feature ($p=1$). The **unregularized** loss is

$$
L(\beta_0,\beta_1)=\frac12\sum_{i=1}^{N}(y_i-\beta_0-x_i\beta_1)^2.
$$

No $L_1$ or $L_2$ penalty appears in this proof. In the regression objectives in the question the intercept $\beta_0$ is not penalized, but this has no effect on the unregularized loss above.

## 2. Matrix derivation

Set

$$
\beta=\begin{bmatrix}\beta_0\\\beta_1\end{bmatrix},\quad
X=\begin{bmatrix}1&x_1\\\vdots&\vdots\\1&x_N\end{bmatrix},\quad
y=\begin{bmatrix}y_1\\\vdots\\y_N\end{bmatrix}.
$$

Then $L(\beta)=\tfrac12\|y-X\beta\|_2^2$. Suppose the feature values are **not all identical**, so the two columns of $X$ are linearly independent. The least-squares minimizer $\widehat\beta=(X^\top X)^{-1}X^\top y$ is unique and satisfies the normal equations

$$
X^\top(y-X\widehat\beta)=0.
$$

Write $d=\beta-\widehat\beta$. Since $y-X\beta=(y-X\widehat\beta)-Xd$, expanding its squared norm gives

$$
\begin{aligned}
L(\beta)
&=L(\widehat\beta)-d^\top X^\top(y-X\widehat\beta)
 +\frac12d^\top X^\top Xd\\
&=L(\widehat\beta)+\frac12d^\top X^\top Xd.
\end{aligned}
$$

The cross term vanishes by the normal equations. The symmetric matrix $X^\top X$ is positive definite under the nonconstant-feature assumption, because $d^\top X^\top Xd=\|Xd\|^2>0$ for every nonzero $d$.

## 3. Why this quadratic is an ellipse

By the spectral theorem, write $X^\top X=Q\operatorname{diag}(a_1,a_2)Q^\top$, where $Q$ is orthogonal and $a_1,a_2>0$. In rotated coordinates $u=Q^\top(\beta-\widehat\beta)$, the level $L(\beta)=L(\widehat\beta)+c$ for $c>0$ becomes

$$
\frac{a_1u_1^2}{2}+\frac{a_2u_2^2}{2}=c,
\qquad\text{or}\qquad
\frac{u_1^2}{2c/a_1}+\frac{u_2^2}{2c/a_2}=1.
$$

This is an ellipse centered at $\widehat\beta$, with axes along the eigenvectors of $X^\top X$ and semiaxis lengths $\sqrt{2c/a_1}$ and $\sqrt{2c/a_2}$. A circle is the special case of equal eigenvalues. Larger $c$ produces larger concentric ellipses.

At $c=0$ the level set is just the minimizing point; below the minimum it is empty. Those are boundary cases, not nondegenerate ellipses.

## 4. What if all $x_i$ are identical?

If $x_i=a$ for every $i$, predictions depend only on $\beta_0+a\beta_1$. The two columns of $X$ are dependent and $X^\top X$ is singular. The minimizers form the line

$$
\beta_0+a\beta_1=\bar y,
$$

and levels above the minimum are pairs of parallel lines, not ellipses. Thus the question's statement needs the usual **full-column-rank assumption**. A single observation also falls in this degenerate case. No such assumption is explicitly stated in the question, so make it visible rather than asserting strict ellipses unconditionally.

## 5. A small example

For $x=[-1,1]$ and $y=[1,3]$, the line $\widehat\beta_0=2$, $\widehat\beta_1=1$ fits both observations exactly. Here $X^\top X=2I$, so

$$
L(\beta)=(\beta_0-2)^2+(\beta_1-1)^2.
$$

Its positive level sets are circles centered at $(2,1)$, a special kind of ellipse. Other feature configurations generally yield ellipses with unequal axes and possibly a rotation.

## What to retain for a concise answer

State that $X$ has full column rank, expand around the least-squares solution, and diagonalize the positive-definite $X^\top X$. Mention the singular-data exception and the point/empty boundary levels.
