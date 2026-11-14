# Assignment 2 — Question 4.3: sparsity and feature selection

**Goal:** give an informal explanation of which regularizer encourages zero coefficients and which one to choose for feature subset selection.

- [Assignment statement](../../../../Assigments/IFT6135_A26__Student_/assignment2/en/assignment2_theory_en.tex)
- [Previous: 4.2 — Squared-error level sets](02-squared-error-level-sets.md)

**Status:** notes only; the Question 4 LaTeX answer remains blank.

## 1. What does sparsity mean here?

For linear regression, the model is

$$
\widehat y=\beta_0+\beta_1x_1+\cdots+\beta_px_p.
$$

A **sparse** coefficient vector has many entries exactly equal to zero. If $\beta_j=0$, feature $x_j$ contributes nothing to this fitted linear predictor, so the nonzero coefficients identify a subset of features. This is model-based selection, not proof that a feature has no causal or predictive relevance in every other model.

In the given regression objectives, the intercept $\beta_0$ is not included in the regularization sum. Feature selection concerns $\beta_1,\ldots,\beta_p$.

## 2. $L_1$ versus $L_2$

- **$L_1$ (lasso):** the penalty $\lambda\sum_{j=1}^p|\beta_j|$ can set coefficients **exactly to zero**. It therefore promotes sparse solutions.
- **$L_2$ (ridge):** the penalty $\lambda\sum_{j=1}^p\beta_j^2$ generally shrinks coefficients toward zero but does not set them exactly to zero. Ridge can reduce overfitting, but it usually retains all features.

The qualification “generally” matters: an $L_2$ solution can have zero coordinates in special cases. Conversely, $L_1$ does not guarantee that any particular coefficient is zero for every dataset or every choice of $\lambda$; in particular, $\lambda=0$ adds no sparsity penalty.

## 3. Geometric intuition

For two **penalized** coefficients, the constrained forms have different shapes:

- $L_1$: $|\beta_1|+|\beta_2|\le t$ forms a diamond, with corners on the coordinate axes.
- $L_2$: $\beta_1^2+\beta_2^2\le t$ forms a disk, with a smooth boundary.

Imagine expanding the unregularized loss contours until one first touches the allowed region. A diamond's corners make contact at an axis more likely, corresponding to one coefficient being exactly zero. A smooth disk has no corners, so contact is typically away from the axes.

This is an **intuition**, not a guarantee for every dataset. Also, the actual two-parameter pair in 4.2 is $(\beta_0,\beta_1)$ and the intercept is unpenalized: its constraint is not a diamond or disk in that plane. The diagram above instead uses two penalized feature coefficients, $\beta_1$ and $\beta_2$.

## 4. Which to choose for feature subset selection?

Choose **$L_1$ (lasso)** because its solution can assign exactly zero coefficients, leaving a subset of features. The amount of selection depends on the regularization strength and the data. When predictors are strongly correlated, the chosen subset can be unstable; an exact zero should not be interpreted as definitive evidence of irrelevance.

## What to retain for a concise answer

State that $L_1$ promotes exact zeros, whereas $L_2$ usually shrinks coefficients without zeroing them. Choose $L_1$ for subset selection and explain it in words; the question does not require a formal proof.
