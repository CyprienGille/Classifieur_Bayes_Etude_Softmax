# Etude de l'impact de Softmax sur un classifieur de Bayes

## Présentation

Ce repository archive un projet de fin de semestre, portant sur la fonction softmax. Pour le sujet complet, voir la page 12 du pdf `Sujets_Projets_FinSemestre.pdf`.

Le but de ce projet était d’étudier la différence entre les fonctions _argmax_ et _softargmax_ (Souvent appelée "softmax") dans le cadre d’un problème de classification, en se servant d'un classifieur de Bayes.

L'idée principale est d'étudier l'évolution du **risque d'erreur global** (défini rigoureusement dans le rapport) selon que l'on utilise _argmax_ ou _softmax_.

Lorsqu'il nous fallait des données d'exemple pour tester nos résultats théoriques, nous avons inventé un dataset de pizzas aux qualités variables.

## Conseils d'usage:

1.  Pour expérimenter avec les classifieurs, modifier le fichier main.py (arguments d'initialisation, print des différents attributs (dont on peut trouver les noms dans les docstring de ProjetSoftmax.py...)

2.  **Le code** (surtout celui de BayesClassifierVect et GradientProjete) **prend tout son sens quand on le regarde conjointement avec le rapport**.

---

On trouvera dans ce repo:

## `Rapport Projet S7 MAM.pdf`

- Le rapport de projet, parfaitement lisible et dans un style didactique. Détaille les démarches, résultats et démonstrations mathématiques, ainsi que la bibliographie.

## `Maths_Du_Projet.ipynb`

- Le latex des différentes notations et démonstrations utilisées dans le rapport.

## `Projet_Softmax.py`

- Le fichier le plus important, contient les définitions des trois classes `BayesClassifierIdeal`, `BayesClassifierVect` et `GradientProjete`, hautement commentées.

## `pizza_exemple.py`

- Implémente l'exemple de classifieur de Bayes dans le cas réel des pizzas de différentes qualités.
- L'exécuter donnera un exemple de classification de deux pizzas de qualités différentes.

## `main.py`

- Donne un exemple d'utilisation des classifieurs de Bayes généralisés classique et vectoriel.
- Contient un If que l'on peut toggle pour activer, à l'éxécution, soit le plotting du risque en 3d et en 2d que l'on a pu observer dans le rapport, soit l'algorithme du gradient projeté.

## `Présentation Projet S7.pdf`

- Le support (slides) de la présentation finale de ce projet.
