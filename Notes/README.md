# Notes

Personal lecture notes, derivations, and study summaries for IFT6135 — Representation Learning.

Add Markdown notes here as the course progresses, organized by lecture or topic.

## Principes de rédaction

Pour les prochaines notes, conserver la rigueur, la profondeur technique et l’organisation existante, avec des explications en **français** et les termes techniques anglais utiles (sauf demande d’une autre langue).

- Commencer par l’objectif et l’intuition, puis un exemple concret pertinent, avant la dérivation générale. Expliquer pourquoi chaque opération est utile.
- Définir les variables, dimensions et hypothèses; montrer l’origine des nombres et les substitutions importantes. Ne pas sauter les transitions nécessaires à la compréhension.
- Distinguer les notions proches, anticiper les questions « pourquoi ? » et relier le raisonnement aux concepts déjà étudiés.
- Garder les notes structurées, complètes et ciblées sur l’exercice : développer les passages difficiles, sans manuel inutilement long ni transcription de conversation.
- Séparer les rôles : les notes enseignent; les réponses LaTeX synthétisent. La remise reste au niveau master, concise et justifiée, avec les preuves ou calculs explicitement demandés.
- Le critère de réussite est de pouvoir reconstruire, expliquer et transférer la solution indépendamment, pas seulement recopier son résultat.

### Conventions confirmées pour l’Assignment 1

- **4.2 :** `MaxPooling(N)` utilise un stride de $N$ et aucun padding. Les convolutions gardent les réglages indiqués dans l’énoncé.
- **4.4 :** utiliser les strides du tableau, malgré la phrase contradictoire annonçant un stride de 1 partout.

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
- [4.2 — Shapes and parameter counts](assignment1/question4/02-shapes-and-parameters.md): dimensions et paramètres, avec la convention officielle de pooling (stride égal à la fenêtre, padding nul).
- [4.3 — Biases before BatchNorm](assignment1/question4/03-biases-before-batchnorm.md): show the bias cancellation, count the removable parameters, and note the inference adjustment.
- [4.4 — Champ réceptif (receptive field)](assignment1/question4/04-receptive-fields.md): intuition, exemple numérique, dérivation des champs et jumps, tableau et distinctions avec les dimensions de sortie.
- [4.5 — Détection de contours (edge detection)](assignment1/question4/05-edge-detection.md): kernels directionnels minimaux, exemples, réponses signées, ReLU explicite, annulation des régions uniformes/horizontales et limites.

Open a Markdown note in VS Code and press **Ctrl+Shift+V** to preview its text and equations.
