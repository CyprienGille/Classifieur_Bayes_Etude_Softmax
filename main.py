from Projet_Softmax import BayesClassifierIdeal, BayesClassifierVect, GradientProjete

bc = BayesClassifierIdeal(nb_classes=3, nb_features=3)
bc2d = BayesClassifierIdeal(nb_classes=2, nb_features=3)
bcv = BayesClassifierVect(nb_classes=3, nb_pts=100)

gpj = GradientProjete(gradient=bcv.gradient, X_init=[0.1, 0.2, 0.7])

if False:  # toggle le plotting du risque

    bc.plot_risk(500)
    # fermez la première figure pour afficher la deuxième

    bc2d.plot_risk(500)


if True:  # toggle la version vectorielle
    gpj.find_maximum(log_interval=20)
