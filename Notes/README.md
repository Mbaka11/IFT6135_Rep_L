# Notes

Personal lecture notes, derivations, and study summaries for IFT6135 — Representation Learning.

Add Markdown notes here as the course progresses, organized by lecture or topic.

## Assignment 1

### Question 1 — Forward pass, backpropagation, and gradient descent

- [1.1 — Forward pass](assignment1/question1/01-forward-pass.md): quick reminders, notation, a tiny example, the worked solution, and guidance on what to submit.
- [1.2 — Binary cross-entropy derivative](assignment1/question1/02-cross-entropy-derivative.md): the BCE/sigmoid chain rule, the requested numerical derivative, and its interpretation.
- [1.3 — Analytical gradients](assignment1/question1/03-analytical-gradients.md): backpropagation, the five gradient formulas, ReLU masking, outer products, and shape checks.
- [1.4 — Numerical gradients](assignment1/question1/04-numerical-gradients.md): expanded chain-rule substitutions, the four parameter gradients, signs, and shape checks.
- [1.5 — One gradient-descent update](assignment1/question1/05-gradient-descent-update.md): every parameter update with learning rate 0.5, exact arithmetic, and simultaneous-update reminders.
- [1.6 — Prediction after the update](assignment1/question1/06-prediction-after-update.md): the full new forward pass, the prescribed exponential approximation, and prediction/loss improvement checks.

### Question 2 — Activation functions

Each note covers one main part and keeps its sub-subquestions together.

- [2.1 — Identity activations and affine composition](assignment1/question2/01-linear-activations.md): **(a)** compose two affine layers; **(b)** collapse the full network and explain its representational capacity.
- [2.2 — Sigmoid and vanishing gradients](assignment1/question2/02-sigmoid-gradient-bound.md): **(a)** maximum derivative; **(b)** explicit chain-rule gradient bound; **(c)** exponential-vanishing condition.
- [2.3 — ReLU, LeakyReLU, and Maxout](assignment1/question2/03-relu-and-variants.md): **(a)** ReLU derivative; **(b)** inactive-neuron weight gradients; **(c)** compare the two alternatives.

### Question 3 — Gradient descent on a quadratic loss

- [3 — Quadratic gradient descent](assignment1/question3/01-quadratic-gradient-descent.md): a from-scratch tutorial on eigenvectors/eigenvalues, illustrated basis changes, transpose versus square, explicit loss derivatives, identity-matrix factorization, induction, convergence, and fully worked examples.

### Question 4 — Convolutional neural networks

- [4.1 — Equivalent convolution](assignment1/question4/01-equivalent-convolution.md): trace input indices to derive the fused kernel, stride, weights, and biases.
- [4.2 — Shapes and parameter counts](assignment1/question4/02-shapes-and-parameters.md): work through each layer, with stated pooling defaults and a brief non-overlapping alternative.
- [4.3 — Biases before BatchNorm](assignment1/question4/03-biases-before-batchnorm.md): show the bias cancellation, count the removable parameters, and note the inference adjustment.

Open a Markdown note in VS Code and press **Ctrl+Shift+V** to preview its text and equations.
