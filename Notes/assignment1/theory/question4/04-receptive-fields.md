# Assignment 1 — Question 4.4 : comprendre le champ réceptif

**Objectif :** déterminer quelle zone de l’image d’entrée peut influencer **un seul élément** à la sortie de chaque couche.

- [Énoncé de l’exercice](../../../Assigments/IFT6135_A26__Student_/assignment1/en/assignment1_theory_en.tex)
- [Précédent : 4.3 — Biais avant BatchNorm](03-biases-before-batchnorm.md)
- [Suivant : 4.5 — Détection de contours](05-edge-detection.md)
- [Connectivité locale dans le cours, page PDF 10](../../../Slides/Lecture_3_convnets.pdf#page=10)

**La partie 4.4 est maintenant remplie en LaTeX : tableau et justification courte, avec les strides confirmés par l’enseignant.**

## 1. Ce qu’on cherche, et pourquoi

Une première convolution lit un petit morceau de l’image. Une convolution plus profonde lit plusieurs réponses de la couche précédente : chacune résume déjà une zone de l’image. Il faut donc remonter ces dépendances jusqu’à l’entrée, au lieu de regarder seulement le dernier kernel.

Cette zone est le **champ réceptif (*receptive field*)**. Elle indique le contexte disponible pour calculer une caractéristique : un détecteur profond peut combiner des motifs provenant d’une région plus large, même si sa carte de sortie est plus petite.

On calcule ici le **champ réceptif théorique spatial** : l’étendue géométrique permise par les fenêtres de l’architecture, pas le gradient effectivement non nul pour une image particulière. La nuance concernant BatchNorm est précisée en section 6.

**Convention officielle :** l’enseignant a confirmé qu’il faut suivre les strides du tableau, et non la phrase annonçant un stride de 1 partout. Le second `MaxPooling(2)` garde donc son **stride 1 explicitement indiqué**; on ne lui applique pas automatiquement la convention du pooling de 4.2.

## 2. Les quantités à ne pas confondre

| Quantité | Ce qu’elle mesure |
| --- | --- |
| **Kernel/window size** $k$ | Nombre de positions lues dans la carte **qui entre dans cette couche**. |
| **Output dimensions** $(H,W,C)$ | Nombre de valeurs produites et leur organisation. Ce n’est pas la zone vue par une valeur. |
| **Stride** $s$ | Déplacement de la fenêtre dans la carte précédente. |
| **Receptive field** $R$ | Taille d’un côté de la zone correspondante dans **l’image d’origine**. |
| **Jump / effective stride** $J$ | Distance, dans l’image d’origine, entre les positions correspondantes de deux sorties voisines. |
| **Padding** | Bordure ajoutée pour placer les fenêtres; elle affecte les dimensions et l’alignement aux bords. |
| **Channels** | Grilles de caractéristiques parallèles. Leur nombre ne multiplie pas la taille spatiale du champ réceptif. |

Les kernels et strides sont les mêmes horizontalement et verticalement. On peut donc calculer un côté $R$, puis écrire un champ $R\times R$. L’hypothèse d’un seul canal simplifie les dépendances entre canaux, pas le raisonnement spatial.

## 3. Exemple concret : les deux premières couches

Pour voir les indices, prenons provisoirement une version **1D, sans padding**, avec

$$
x=(0,1,2,\ldots,13).
$$

Une convolution de taille 5 et stride 3 utilise un filtre dont les cinq coefficients valent 1. Ce choix sert seulement à rendre les sorties calculables :

| Sortie de convolution | Positions utilisées | Valeur |
| --- | --- | ---: |
| $a_0$ | $x_0,\ldots,x_4$ | $0+1+2+3+4=10$ |
| $a_1$ | $x_3,\ldots,x_7$ | $3+4+5+6+7=25$ |
| $a_2$ | $x_6,\ldots,x_{10}$ | $6+7+8+9+10=40$ |
| $a_3$ | $x_9,\ldots,x_{13}$ | $9+10+11+12+13=55$ |

Chaque $a_i$ voit 5 positions d’entrée, donc $R=5$. Deux fenêtres voisines commencent à 0 et 3, donc $J=3$.

Le max pooling de taille 2 et stride 2 produit

$$
p_0=\max(a_0,a_1)=25,\qquad
p_1=\max(a_2,a_3)=55.
$$

Pour $p_0$, les deux candidats couvrent les indices $[0,4]$ et $[3,7]$. Leur union est $[0,7]$, soit

$$
R=7-0+1=8.
$$

**Pourquoi pas 2 ou 10 ?** La fenêtre de pooling lit deux *activations*, pas deux pixels d’origine. Chaque activation voit cinq pixels, mais leurs zones se recouvrent : additionner $5+5$ compterait deux pixels deux fois.

Le champ de $p_1$ couvre $[6,13]$. Les débuts des deux champs sont donc espacés de $6-0=6$ pixels : le nouveau $J$ vaut 6.

On a seulement **deux sorties de pooling**, mais chacune possède un champ réceptif de **huit positions**. Ce sont deux mesures différentes.

Pour ces valeurs, $a_1$ gagne le premier max. Le champ théorique considère néanmoins les deux candidats : sur une autre entrée, $a_0$ pourrait gagner. Il ne faut pas confondre cette étendue avec le chemin du gradient pour cet exemple précis.

## 4. Dériver la règle générale

À la couche $\ell$, on note :

- $k_\ell$ : taille de la fenêtre;
- $s_\ell$ : stride de cette couche;
- $R_{\ell-1}$ : champ réceptif d’un élément de la couche précédente;
- $J_{\ell-1}$ : espacement de deux éléments précédents, mesuré dans l’entrée d’origine.

Au départ, chaque pixel d’entrée voit seulement lui-même et les pixels voisins sont espacés d’une position :

$$
R_0=1,\qquad J_0=1.
$$

### Taille du champ

Une fenêtre lit $k_\ell$ éléments précédents. Entre le premier et le dernier, il y a **$k_\ell-1$ intervalles**, chacun de longueur $J_{\ell-1}$ dans l’image d’origine. Le premier élément couvre déjà une largeur $R_{\ell-1}$; le dernier prolonge cette zone de $(k_\ell-1)J_{\ell-1}$. Ainsi,

$$
\boxed{R_\ell=R_{\ell-1}+(k_\ell-1)J_{\ell-1}.}
$$

### Espacement des sorties

Pour produire la sortie voisine, la nouvelle fenêtre avance de $s_\ell$ positions dans la carte précédente. Chaque position correspond à $J_{\ell-1}$ pixels d’origine :

$$
\boxed{J_\ell=s_\ell J_{\ell-1}.}
$$

**Point essentiel :** le calcul de $R_\ell$ utilise l’**ancien** $J_{\ell-1}$, pas le nouveau $J_\ell$. Le stride actuel déplace la fenêtre entre deux sorties; il n’espace pas davantage les éléments lus à l’intérieur d’une seule fenêtre.

Ces règles valent pour convolution et pooling, sans dilation. BatchNorm conserve le champ géométrique et le jump dans la convention expliquée plus bas.

## 5. Appliquer les règles au tableau

Les deux premières couches ont déjà donné $R=8$, $J=6$. Après BatchNorm, `Conv(4,1)` lit quatre activations espacées de 6 pixels d’origine. Il y a trois intervalles entre leurs débuts :

$$
R=8+(4-1)\cdot6=8+18=26.
$$

Son stride 2 donne séparément $J=2\cdot6=12$. Le calcul suivant utilise maintenant ce jump de 12.

| Couche | $k$ | $s$ | Calcul du côté $R$ | Calcul de $J$ | Champ réceptif |
| --- | ---: | ---: | --- | --- | --- |
| Entrée | — | — | $1$ | $1$ | $1\times1$ |
| Conv(5,1), stride 3 | 5 | 3 | $1+(5-1)\cdot1=5$ | $3\cdot1=3$ | $5\times5$ |
| MaxPooling(2), stride 2 | 2 | 2 | $5+(2-1)\cdot3=8$ | $2\cdot3=6$ | $8\times8$ |
| BatchNorm | — | — | $8$ | $6$ | $8\times8$ |
| Conv(4,1), stride 2 | 4 | 2 | $8+(4-1)\cdot6=26$ | $2\cdot6=12$ | $26\times26$ |
| MaxPooling(2), stride 1 | 2 | 1 | $26+(2-1)\cdot12=38$ | $1\cdot12=12$ | $38\times38$ |
| Conv(3,1), stride 1 | 3 | 1 | $38+(3-1)\cdot12=62$ | $1\cdot12=12$ | $62\times62$ |

Le dernier $62\times62$ décrit la zone vue par **une activation finale**, pas les dimensions de la carte de sortie. Ces dernières demanderaient aussi la taille de l’entrée et le padding, comme en [4.2](02-shapes-and-parameters.md).

## 6. BatchNorm, padding et portée du résultat

**BatchNorm.** Avec des statistiques fixées, notamment en mode *inference*, BatchNorm transforme chaque valeur séparément, sans lire un nouveau voisinage spatial : $R$ et $J$ restent inchangés. En mode *training*, les statistiques calculées sur le batch et les positions spatiales introduisent des dépendances globales. Le tableau utilise le champ géométrique usuel des fenêtres locales, pas ce graphe complet de dépendances pendant l’entraînement. C’est la même distinction training/inference rencontrée en [4.3](03-biases-before-batchnorm.md).

**Padding.** Il déplace l’origine des fenêtres, mais ne change ni les distances entre leurs positions ni leur étendue nominale. C’est pourquoi il n’apparaît pas dans les deux récurrences. Près du bord, une partie du champ peut couvrir du padding plutôt que de vrais pixels; le tableau décrit la taille théorique, pas le nombre de pixels réels disponibles à chaque bord.

## 7. Connexion à 4.1 et auto-vérification

En [4.1](01-equivalent-convolution.md), les paires $(k,s)$ étaient $(3,3)$, $(2,2)$ et $(2,2)$. Les mêmes règles donnent

$$
(R,J):(1,1)\longrightarrow(3,3)\longrightarrow(6,6)\longrightarrow(12,12).
$$

On retrouve la taille et le stride de la convolution fusionnée. Mais **connaître le champ ne prouve pas une équivalence entre opérations** : 4.1 permet une fusion grâce à la linéarité des convolutions et de l’average pooling. Le max pooling de 4.4 est non linéaire; un champ de $62\times62$ ne permet donc pas de remplacer automatiquement ce réseau par une convolution $62\times62$.

**Petit transfert :** si le premier pooling devenait `MaxPooling(3)`, stride 1, que changerait-il juste après la première convolution ? En repartant de $R=5$, $J=3$,

$$
R=5+(3-1)\cdot3=11,\qquad J=1\cdot3=3.
$$

Le champ grandirait à $11\times11$, mais le jump resterait 3 : une fenêtre plus large et un déplacement plus grand ne sont pas la même chose.

## Pour la réponse courte

Le tableau LaTeX contient les six champs réceptifs. La réponse rappelle les deux règles, illustre le calcul de Conv(4,1) et précise que BatchNorm conserve le champ géométrique. Les exemples et la dérivation détaillée restent dans cette note; **4.4 est désormais remplie en LaTeX**.
