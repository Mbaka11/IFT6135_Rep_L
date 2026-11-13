# Assignment 1 — Question 4.2 : dimensions et paramètres

**Objectif :** suivre ce que chaque couche transforme, puis distinguer la taille des données produites du nombre de valeurs apprises.

- [Énoncé de l’exercice](../../../../Assigments/IFT6135_A26__Student_/assignment1/en/assignment1_theory_en.tex)
- [Précédent : 4.1 — Convolution équivalente](01-equivalent-convolution.md)
- [Suivant : 4.3 — Biais supprimables](03-biases-before-batchnorm.md)
- [Pooling dans le cours, page PDF 47](../../../../Slides/Lecture_3_convnets.pdf#page=47)

**La réponse 4.2 est remplie en LaTeX et corrigée selon la clarification de l’enseignant : `MaxPooling(N)` utilise un stride de $N$ et aucun padding.** L’ancienne lecture avec padding 1/stride 1 pour le pooling n’est plus à retenir.

## Rappels rapides

- Un **channel** est une grille de valeurs. L’entrée RGB possède trois canaux; une convolution produit des **feature maps**, qui représentent des caractéristiques apprises.
- Un **kernel** est une fenêtre de coefficients. `Conv(4,8)` signifie des fenêtres $4\times4$ et **8 canaux de sortie**, pas 8 pixels par côté.
- Le **stride** est le déplacement de la fenêtre. Le **padding** ajoute une bordure : padding 1 ajoute une position de chaque côté, donc deux par dimension.
- Le **max pooling** garde la plus grande valeur d’une fenêtre dans chaque canal. Ici, les fenêtres ne se recouvrent pas : leur stride égale leur taille. Il réduit la résolution sans apprendre de coefficients.
- Un **paramètre** est une valeur ajustée pendant l’apprentissage. Une **activation** est une valeur calculée pour une entrée. Multiplier le nombre d’activations ne multiplie pas forcément les paramètres.

## 1. Fixer les conventions

Pour les convolutions, on conserve le padding $P=1$. Le premier stride est explicitement $S=2$; le second reste $S=1$.

Pour `MaxPooling(2)`, on utilise $K=2$, $P=0$, $S=2$. Pour `MaxPooling(4)`, on utilise $K=4$, $P=0$, $S=4$, conformément à la clarification officielle.

Les dimensions sont notées $(H,W,C)$, sans la dimension de batch. Les deux dernières lignes utilisent $(1,1,D)$ pour respecter le format du tableau : il s’agit d’un vecteur de $D$ caractéristiques, pas d’une image spatiale.

## 2. Comprendre les trois formules

### Nombre de positions de sortie

Une fenêtre doit tenir dans une dimension de longueur $N_{\mathrm{in}}+2P$. Son premier départ est à 0; le dernier départ autorisé est $N_{\mathrm{in}}+2P-K$. Comme on avance de $S$ positions, on obtient

$$
N_{\mathrm{out}}=
\left\lfloor\frac{N_{\mathrm{in}}+2P-K}{S}\right\rfloor+1.
$$

Le $+1$ compte la première fenêtre. Le floor arrondit vers le bas quand une dernière fenêtre ne tient pas entièrement. On applique cette formule séparément à la hauteur et à la largeur.

### Paramètres d’une convolution

Chaque canal de sortie utilise $K^2$ coefficients **par canal d’entrée**, et un biais. Ainsi,

$$
\#\mathrm{Conv}=(K^2C_{\mathrm{in}}+1)C_{\mathrm{out}}.
$$

Les mêmes coefficients sont réutilisés à toutes les positions (*weight sharing*). On ne multiplie donc pas ce compte par la hauteur et la largeur de sortie.

### Paramètres d’une couche dense

Dans une couche **fully connected (FC)**, chacun des $D_{\mathrm{out}}$ neurones utilise les $D_{\mathrm{in}}$ entrées et un biais :

$$
\#\mathrm{FC}=(D_{\mathrm{in}}+1)D_{\mathrm{out}}.
$$

**BatchNorm** conserve la forme et apprend une échelle $\gamma_c$ et un décalage $\beta_c$ par canal : $2C$ paramètres. Ses moyennes et variances stockées sont des statistiques, pas des paramètres entraînables. ReLU, pooling et flattening n’en ont aucun.

## 3. Suivre les couches dans l’ordre

### Première convolution : Conv(4,8), stride 2

L’entrée est $(64,64,3)$. Avec $K=4$, $P=1$, $S=2$,

$$
N_{\mathrm{out}}=
\left\lfloor\frac{64+2-4}{2}\right\rfloor+1=31+1=32.
$$

Les 8 canaux produits donnent $(32,32,8)$. Chaque filtre a $4^2\cdot3=48$ coefficients, puis un biais :

$$
(48+1)\cdot8=392.
$$

BatchNorm a $2\cdot8=16$ paramètres. BatchNorm et ReLU gardent la forme $(32,32,8)$.

### MaxPooling(2)

Sur une dimension de longueur 32, les fenêtres de taille 2 commencent aux positions $0,2,\ldots,30$ : il y en a 16.

$$
N_{\mathrm{out}}=
\left\lfloor\frac{32-2}{2}\right\rfloor+1=15+1=16.
$$

La sortie est $(16,16,8)$, avec aucun paramètre. Les canaux sont conservés : on regroupe des positions **dans** chaque grille, sans mélanger les grilles.

### Deuxième convolution : Conv(3,16)

Elle reçoit maintenant 8 canaux et une grille $16\times16$. Avec $K=3$, $P=1$, $S=1$,

$$
N_{\mathrm{out}}=\frac{16+2-3}{1}+1=16,
$$

$$
\#\mathrm{Conv}=(3^2\cdot8+1)\cdot16=(72+1)\cdot16=1168.
$$

La sortie est $(16,16,16)$. Le deuxième 16 est une dimension spatiale; le troisième est le nombre de canaux. Ils sont égaux ici par hasard. BatchNorm ajoute $2\cdot16=32$ paramètres; BatchNorm et ReLU ne changent pas la forme.

### MaxPooling(4)

Les fenêtres commencent aux positions $0,4,8,12$ dans une dimension de longueur 16 :

$$
N_{\mathrm{out}}=
\left\lfloor\frac{16-4}{4}\right\rfloor+1=3+1=4.
$$

La sortie est $(4,4,16)$, sans paramètre.

### Flatten et FC(10)

Le **flattening** conserve toutes les valeurs, mais les range dans un vecteur :

$$
D_{\mathrm{in}}=4\cdot4\cdot16=256.
$$

Il ne change ni les valeurs ni leur nombre. La couche FC apprend à combiner ces 256 caractéristiques pour produire 10 sorties :

$$
\#\mathrm{FC}=(256+1)\cdot10=2570.
$$

## 4. Tableau corrigé : convention confirmée par l’enseignant

| Couche | Dimensions de sortie $(H,W,C)$ | Paramètres entraînables |
| --- | --- | ---: |
| Input | $(64,64,3)$ | 0 |
| Conv(4,8), stride 2 | $(32,32,8)$ | 392 |
| BatchNorm | $(32,32,8)$ | 16 |
| ReLU | $(32,32,8)$ | 0 |
| MaxPooling(2), stride 2, padding 0 | $(16,16,8)$ | 0 |
| Conv(3,16) | $(16,16,16)$ | 1168 |
| BatchNorm | $(16,16,16)$ | 32 |
| ReLU | $(16,16,16)$ | 0 |
| MaxPooling(4), stride 4, padding 0 | $(4,4,16)$ | 0 |
| Reshape | $(1,1,256)$ | 0 |
| FC(10) | $(1,1,10)$ | 2570 |
| **Total** | — | **4178** |

Le total vient de $392+16+1168+32+2570=4178$. Les biais des convolutions sont inclus : 4.2 compte les couches données; [4.3](03-biases-before-batchnorm.md) explique ensuite lesquels sont redondants.

## 5. Vérifications conceptuelles

- **Pourquoi le pooling change-t-il le compte de FC, mais pas celui des convolutions ?** FC possède un poids par entrée; une grille plus petite donne moins d’entrées. Les convolutions partagent leurs coefficients entre positions : à kernel et canaux identiques, leur compte ne dépend pas de la résolution.
- **Pourquoi ne pas supprimer les biais dès ce tableau ?** Cela mélangerait le compte de l’architecture donnée avec sa simplification demandée en 4.3.
- **Que faut-il mettre dans la remise ?** Le tableau et les calculs utiles pour vérifier dimensions et paramètres. Les rappels pédagogiques restent dans les notes.
