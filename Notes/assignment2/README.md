# Assignment 2 — Notes et programmation

L’espace de travail est prêt et les énoncés théorique et pratique sont disponibles en anglais et en français. La question théorique 1 est remplie; les autres questions restent à faire.

- [Documents et rapports](../../Assigments/IFT6135_A26__Student_/assignment2/) : dossiers `en/` et `fr/`.
- [Notes théoriques](theory/) : organisées par question.
- [Code Python](python-code/) : fichiers de départ et implémentations.
- [Notebooks](python-code/notebooks/) : notebooks de travail et versions exécutées.

Follow the [writing principles](../README.md#principes-de-rédaction): notes explain difficult steps; LaTeX answers stay concise and justified. **Write Assignment 2 notes in English**, matching the English submission. Add the Python starter files before beginning the practical work; do not invent results.

Les quatre documents LaTeX compilent avec la recette VS Code **MiKTeX (local)** (`Ctrl+Alt+B`). Les PDF sont générés dans les dossiers `en/latex-build/` et `fr/latex-build/`, ignorés par Git; `Ctrl+Alt+V` ouvre le PDF du document actif.

Les corrections portent sur les préambules : chemin `../../math_commands.tex`, Babel `french`, dépendances manquantes et configuration des en-têtes/environnements. Un modèle de réponse a été ajouté pour la question théorique 1 : les parties 1.1 à 1.4 sont remplies, y compris l’algorithme demandé en 1.2 et le tableau de complexité; les autres questions restent à faire.

## Théorie — Question 1

- [1.1 — Rewriting linear attention](theory/question1/01-attention-lineaire.md): Q/K/V reminders, a worked example, numerator and denominator factorization, dimensions, and the normalization condition.
- [1.2 — Recurrent form](theory/question1/02-recurrent-form.md): state recurrence, initialization, algorithm order, dimensions, and a two-step example.
- [1.3 — Parallel form](theory/question1/03-parallel-form.md): row stacking, score matrix, causal masking, row-wise proof, and a three-position example.
- [1.4 — Comparing complexity](theory/question1/04-complexity.md): total work, sequential-stage assumptions, a numerical example and parallel-prefix caveats.

## Theory — Question 2

- [2.1 — Counting sequences](theory/question2/01-counting-sequences.md): exact count, fixed start token and vocabulary assumptions.
- [2.2 — Exhaustive enumeration](theory/question2/02-exhaustive-enumeration.md): candidate growth, prefix sharing and scalability.
- [2.3 — Beam search and complexity](theory/question2/03-beam-search-complexity.md): worked example, selection and memory costs, and algorithm assumptions.
- [2.4 — Greedy and beam decoding](theory/question2/04-greedy-and-beam-decoding.md): step-by-step tree trace and initialization caveat.
- [2.5 — Decoding limitations](theory/question2/05-decoding-limitations.md): why local choices and beam pruning can miss better sequences.

## Theory — Question 3

- [3.1 — Effect of temperature](theory/question3/01-effect-of-temperature.md): weights and limits, softmax derivatives, and context-vector interpretation.
- [3.2 — Scaling alignment scores](theory/question3/02-scaling-scores.md): scale–temperature invariance and selectivity counterexamples.
- [3.3 — Adaptive temperature](theory/question3/03-adaptive-temperature.md): shared versus state-dependent temperature and why temperature alone does not determine attention.

## Theory — Question 4

- [4.1 — Weight decay and L2](theory/question4/01-weight-decay-and-l2.md): SGD, RMSProp and Adam updates and equivalence assumptions.
- [4.2 — Squared-error level sets](theory/question4/02-squared-error-level-sets.md): ellipse derivation and rank-deficient cases.
- [4.3 — Sparsity and feature selection](theory/question4/03-sparsity-and-feature-selection.md): L1 versus L2 and geometric intuition.

## Theory — Question 5

- [5.1 — ViT summary](theory/question5/01-summary.md): architecture, pretraining and reported results.
- [5.2 — Strengths](theory/question5/02-strengths.md): conceptual and experimental contributions.
- [5.3 — Weaknesses](theory/question5/03-weaknesses.md): data, compute and evaluation limitations.
- [5.4 — Reflections](theory/question5/04-reflections.md): course connections and a testable follow-up experiment.

Conserver le code, les notebooks et les petits résumés CSV/JSON dans Git. Les datasets, checkpoints, tableaux NumPy générés et graphiques d’expériences redondants restent locaux; les figures utilisées dans un rapport sont conservées avec celui-ci.
