# Assignment 1 — Question 4.5 : construire des filtres de contours

**Objectif :** construire deux kernels $3\times3$ qui signalent les deux sens d’une transition verticale, sans réagir aux régions uniformes ni aux transitions purement horizontales.

- [Énoncé de l’exercice](../../../Assigments/IFT6135_A26__Student_/assignment1/en/assignment1_theory_en.tex)
- [Précédent : 4.4 — Champ réceptif](04-receptive-fields.md)
- [Rappel : ReLU et activations](../question2/03-relu-and-variants.md)

**La partie 4.5 est maintenant remplie en LaTeX.** La réponse distingue la réponse signée du kernel de la sortie après **ReLU**. Cette activation reste une étape supplémentaire, non explicitement donnée dans l’énoncé et signalée comme telle.

## 1. Comprendre l’orientation avant de choisir les coefficients

Les pixels valent 0 pour le noir (*dark*) et 1 pour le blanc (*light*). Nous cherchons un **contraste**, pas simplement une présence de blanc : une zone entièrement blanche ne contient pas de contour.

Un **bord vertical (*vertical edge*)** s’étend de haut en bas, mais la luminosité change quand on le traverse **de gauche à droite**. Il faut donc comparer des pixels horizontalement. Un bord horizontal demande au contraire une comparaison haut/bas.

Les deux détecteurs séparent la **polarité** du contraste :

- *dark-to-light* : noir à gauche, blanc à droite;
- *light-to-dark* : blanc à gauche, noir à droite.

## 2. Un exemple simple pour découvrir la différence utile

Regardons trois pixels d’une ligne. Comparer la droite à la gauche donne :

| Ligne observée | Droite moins gauche | Interprétation |
| --- | --- | --- |
| $(0,0,1)$ | $1-0=1$ | Noir vers blanc |
| $(0,1,1)$ | $1-0=1$ | Même transition, décalée dans la fenêtre |
| $(1,1,0)$ | $0-1=-1$ | Blanc vers noir |
| $(1,0,0)$ | $0-1=-1$ | Même transition inverse, décalée |
| $(0,0,0)$ | $0-0=0$ | Région noire uniforme |
| $(1,1,1)$ | $1-1=0$ | Région blanche uniforme |

La différence garde l’information de direction dans son **signe**. Additionner les pixels ne le ferait pas : une région blanche donnerait une grande somme, même sans contour.

Pour réaliser cette différence, le coefficient de gauche doit être $-1$, celui de droite $+1$. Le coefficient central peut être 0 : les deux placements possibles du bord ont les mêmes valeurs aux extrémités. Aucun facteur de normalisation n’est nécessaire, puisque $1-0$ vaut déjà exactement 1.

## 3. Transformer cette idée en deux kernels

### Convention et dimensions

Soit $X$ un **patch** $3\times3$ de l’image. $X_{ij}$ désigne son pixel à la ligne $i$, colonne $j$, avec $i,j\in\{1,2,3\}$. Chaque kernel produit **une valeur** par patch, pas une matrice $3\times3$.

Comme précédemment, on utilise la convention CNN sans retournement du kernel (*cross-correlation*). La réponse brute est une somme de produits **position par position**, et non le produit matriciel $KX$ :

$$
z_K(X)=\sum_{i=1}^{3}\sum_{j=1}^{3}K_{ij}X_{ij}.
$$

On choisit un biais nul. Avec une convolution mathématique qui retourne le kernel de $180^\circ$, il faudrait intervertir les noms des deux filtres ci-dessous.

### Kernels minimaux

Plaçons la comparaison sur la ligne centrale et mettons à zéro les positions inutilisées :

$$
K_+=\begin{bmatrix}
0&0&0\\
-1&0&1\\
0&0&0
\end{bmatrix},
\qquad
K_-=-K_+=\begin{bmatrix}
0&0&0\\
1&0&-1\\
0&0&0
\end{bmatrix}.
$$

$K_+$ vise le sens noir-vers-blanc; $K_-$ vise le sens blanc-vers-noir. Tous leurs coefficients appartiennent à $\{-1,0,1\}$, donc respectent la contrainte de l’énoncé. Chacun n’a que deux coefficients non nuls.

En ne gardant que ces deux termes dans la somme,

$$
z_+(X)=-X_{2,1}+X_{2,3},\qquad
z_-(X)=X_{2,1}-X_{2,3}=-z_+(X).
$$

Par exemple, pour le bord vertical

$$
X=\begin{bmatrix}0&0&1\\0&0&1\\0&0&1\end{bmatrix},
$$

on obtient $z_+=-0+1=1$ et $z_-=0-1=-1$. En inversant noir et blanc, les réponses s’inversent.

**Pourquoi seulement deux coefficients ?** Pour annuler une région blanche, leur somme doit être nulle. Un filtre non nul avec un seul coefficient ne peut pas satisfaire cette condition. Deux coefficients opposés suffisent; utiliser les trois lignes est possible, mais n’est pas nécessaire pour les bords idéaux de l’exercice.

## 4. Obtenir 0 sur la direction inverse : le rôle explicite de ReLU

Le second kernel ne donne pas encore 0 sur l’exemple précédent : il donne $-1$. On définit donc les sorties des deux détecteurs par

$$
h_+(X)=\max(0,z_+(X)),\qquad
h_-(X)=\max(0,z_-(X)).
$$

ReLU conserve la réponse positive de la direction voulue et supprime la réponse négative de l’autre direction. Contrairement à une valeur absolue, elle ne rend pas les deux polarités indistinguables.

Notons $g=X_{2,1}$ le pixel gauche et $d=X_{2,3}$ le pixel droit. Puisqu’ils sont binaires, les quatre cas sont :

| $g$ | $d$ | $z_+=d-g$ | $z_-=g-d$ | $h_+$ | $h_-$ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 1 | 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | −1 | 1 | 0 |
| 1 | 0 | −1 | 1 | 0 | 1 |

Les sorties après ReLU sont donc exactement 0 ou 1, pour **tout patch binaire**, selon le contraste entre les deux pixels sélectionnés. Cela ne signifie pas que tout patch activant un filtre possède un contour exclusivement vertical; voir les limites en section 6.

### Pourquoi un autre kernel linéaire ne résout-il pas seul le problème ?

C’est une limite de la linéarité, pas un mauvais choix de coefficients. Exiger une réponse nulle sur le patch entièrement noir impose un biais nul. Exiger aussi zéro sur le patch entièrement blanc $U$ impose $z_K(U)=0$.

Si un bord $E$ donne $z_K(E)=1$, son complément noir/blanc $U-E$ donne nécessairement

$$
z_K(U-E)=z_K(U)-z_K(E)=0-1=-1.
$$

Il ne peut donc pas donner 0 sans étape supplémentaire. La ReLU permet cette suppression; **elle ne fait pas partie du kernel lui-même**. Si la consigne devait interdire toute activation, son exigence littérale 0/1 serait irréalisable pour ces deux polarités et les régions uniformes.

## 5. Pourquoi les régions uniformes et les bords horizontaux s’annulent

### Région uniforme

Si tous les pixels du patch valent $c$, avec $c=0$ ou $c=1$,

$$
z_+=-c+c=0,\qquad z_-=0.
$$

La somme nulle des coefficients retire la composante constante de l’image : les filtres répondent à une différence, pas au niveau de luminosité.

### Bord horizontal

Prenons du noir en haut et du blanc en bas :

$$
X_H=\begin{bmatrix}0&0&0\\0&0&0\\1&1&1\end{bmatrix}.
$$

Sur la ligne centrale, gauche et droite valent tous deux 0, donc les deux réponses sont nulles. Sur un bord horizontal déplacé, ils pourraient tous deux valoir 1; la différence resterait nulle.

Plus généralement, un patch constant le long de chaque ligne a la forme $X_{ij}=c_i$. Chaque ligne de nos kernels a une somme nulle, d’où

$$
z_K(X)=\sum_i c_i\left(\sum_j K_{ij}\right)=0.
$$

**Deux propriétés distinctes :** une somme totale nulle annule les régions uniformes; une somme nulle **sur chaque ligne** annule aussi les changements de luminosité haut/bas. Les coefficients opposés gauche/droite sélectionnent ainsi la variation horizontale qui traverse un contour vertical. ReLU sélectionne ensuite sa polarité.

## 6. Localisation et limites à ne pas masquer

- **« Autour du bord » n’est pas « sur une seule colonne ».** Sur la ligne $(0,0,0,1,1,1)$, les fenêtres de taille 3 sont $(0,0,0)$, $(0,0,1)$, $(0,1,1)$, $(1,1,1)$. Leurs réponses $z_+$ sont $(0,1,1,0)$. Le bord active deux positions voisines parce qu’on compare des pixels espacés de deux colonnes.
- **Ce n’est pas un classificateur parfait d’orientations.** Un patch diagonal, par exemple $\begin{bmatrix}0&0&1\\0&1&1\\1&1&1\end{bmatrix}$, possède lui aussi $X_{2,3}-X_{2,1}=1$. Il active $K_+$. Un bord oblique peut avoir une composante de contraste horizontal; on garantit l’annulation des bords purement horizontaux, pas de tout autre motif imaginable. Inversement, une ligne $(0,1,0)$ donne 0 : les deux contrastes opposés se compensent entre les extrémités.
- **Répéter la ligne change l’échelle.** Trois lignes $[-1,0,1]$ donnent $1+1+1=3$ sur le bord vertical idéal. Diviser ce filtre par 3 ramène cette réponse à 1, mais des patches partiels peuvent produire $1/3$ ou $2/3$ après ReLU. Le filtre minimal évite ce changement d’échelle et conserve des sorties binaires.
- **Les bords de l’image sont un cas séparé.** Nos exemples concernent des patches intérieurs. Un zero padding peut créer un contraste artificiel avec une région blanche au bord de l’image.

Ainsi, « 0 ailleurs » est satisfait pour les régions uniformes, la polarité opposée après ReLU et les bords purement horizontaux. Il ne faut pas le présenter comme une garantie universelle sur tous les motifs binaires non verticaux.

## 7. Connexion à 4.4 et petit transfert

Le format $3\times3$ fixe la fenêtre disponible, mais nos zéros font que seuls deux pixels contribuent réellement. Cela illustre la distinction de [4.4](04-receptive-fields.md) entre champ géométrique de l’architecture et dépendances effectivement utilisées.

**À refaire pour vérifier l’idée :** que détecterait le kernel transposé $K_+^\top$ ? Les coefficients gauche/droite deviennent haut/bas :

$$
z_{K_+^\top}(X)=-X_{1,2}+X_{3,2}.
$$

Il donne 1 sur $X_H$ : la luminosité augmente du haut vers le bas, donc le contour détecté est horizontal. Transposer le kernel change l’orientation recherchée; le nier change seulement la polarité.

## Pour la réponse courte

La réponse LaTeX présente les deux kernels, leurs réponses $z_+$ et $z_-$ et **la ReLU explicitement ajoutée** pour obtenir 0/1. Elle justifie les signes opposés et l’annulation des régions uniformes et des bords horizontaux, sans promettre le rejet de toutes les orientations obliques. Les exemples et la preuve de la limite linéaire restent dans cette note.
