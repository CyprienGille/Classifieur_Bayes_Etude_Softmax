from Projet_Softmax import BayesClassifierIdeal


## Les probabilités arbitraires
pizza_ranges = [[1, 6], [0, 30, 5], [1, 5], [10, 25, 5]]
pizza_probas = [
    [[0.0, 0.1, 0.8, 0.1, 0.0], [0.0, 0.3, 0.4, 0.3, 0.0], [0.4, 0.1, 0.0, 0.1, 0.4]],
    [
        [0.0, 0.6, 0.4, 0.0, 0.0, 0.0],
        [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
        [0.3, 0.0, 0.0, 0.0, 0.1, 0.6],
    ],
    [[0.0, 0.0, 0.5, 0.5], [0.0, 0.45, 0.45, 0.1], [0.6, 0.40, 0.0, 0.0]],
    [[0.1, 0.4, 0.5], [0.3, 0.4, 0.3], [0.89, 0.1, 0.01]],
]
pizza_prop = [0.3, 0.5, 0.2]


## classifieur de Bayes pour les pizzas
pizza_bc = BayesClassifierIdeal(
    nb_classes=3, nb_features=4, feature_ranges=pizza_ranges
)
pizza_bc.probas = pizza_probas
pizza_bc.prop = pizza_prop


print("Une bonne pizza")
miam = [3, 5, 3, 20]
pizza_bc.classify(miam)


print("\nClassification d'une pizza la plus mauvaise possible")
mauvaise_pizza = [1, 0, 1, 10]
pizza_bc.classify(mauvaise_pizza)
pizza_bc.bayes_softmax(mauvaise_pizza)
