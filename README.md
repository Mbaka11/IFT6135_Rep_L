# IFT6135 — Representation Learning

Course materials and personal study work for **IFT6135 — Representation Learning**, collected during my master's studies in machine learning at **Mila**.

## Repository structure

```text
IFT6135_Rep_L/
├── Assigments/
│   └── IFT6135_A26__Student_/
│       ├── assignment0/        # Assignment statements and supplied solutions
│       ├── assignment1/        # Statements, theory answers and practical report
│       ├── assignment2/        # English/French theory and practical handouts
│       └── math_commands.tex   # Shared LaTeX notation
├── Slides/                     # Lecture slides and supplementary presentations
└── Notes/
    ├── assignment1/            # theory/ notes and python-code/ workspace
    └── assignment2/            # Theory and programming folders ready for use
```

The existing `Assigments` folder name is preserved. Assignment materials include English and French LaTeX sources.

## Topics

- Neural networks, multilayer perceptrons, and convolutional networks
- Optimization, training, and regularization
- Recurrent networks and transformers
- Autoencoders, variational autoencoders, GANs, and generative modeling
- Autoregressive models, normalizing flows, and diffusion models
- Self-supervised learning and meta-learning

## Assignments

| Assignment | Materials | Personal work |
| --- | --- | --- |
| [Assignment 0](Assigments/IFT6135_A26__Student_/assignment0/) | Theory statements and supplied solution PDFs | Not started in this repository |
| [Assignment 1](Assigments/IFT6135_A26__Student_/assignment1/) | Theory and programming statements covering MLPs and CNNs | [Study notes](Notes/README.md#assignment-1), [Python work](Notes/assignment1/python-code/README.md), and [practical report](Assigments/IFT6135_A26__Student_/assignment1/en/assignment1_practical_report_en.pdf) |
| [Assignment 2](Assigments/IFT6135_A26__Student_/assignment2/) | Theory and practical handouts in English and French | [Notes and programming workspace](Notes/assignment2/README.md) prepared; answers not started |

## Working conventions

- Keep statements and reports in `Assigments/IFT6135_A26__Student_/assignmentN/`, with `en/` and `fr/` language folders.
- Keep explanations in `Notes/assignmentN/theory/` and code/notebooks in `Notes/assignmentN/python-code/`.
- Track source code, executed notebooks, compact CSV/JSON result summaries, and figures required by the reports.
- Keep datasets, model arrays/checkpoints, redundant experiment plots, result ZIPs, virtual environments and credentials out of Git. Ignored files remain available locally.
- Lecture PDFs and report PDFs are tracked; LaTeX build intermediates stay in ignored `latex-build/` folders.

Assignment 1's pinned Python dependencies are listed in [requirements.txt](Notes/assignment1/python-code/student/requirements.txt). Assignment 2 dependencies will be determined from its handout. Worked solutions are kept in this **private** repository.

## Attribution and course policy

This is a personal study repository, not an official course resource. Course slides, assignment statements, and supplied solutions belong to their respective authors and are not relicensed here. Some assignment documents are marked **“Do not distribute”**; follow the course's redistribution and academic-integrity policies. Public availability does not grant permission to reuse or redistribute these materials.
