import numpy as np
from numpy.random import default_rng
from numpy import argmax
from scipy.special import softmax
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


class BayesClassifierIdeal:
    """Cette classe implémente tout le processus d'un classifieur de bayes avec connaissance idéale
    des probabilités
    
    """

    def __init__(
        self,
        nb_classes: int = 3,
        nb_features: int = 4,
        use_softmax: bool = False,
        seed: int = None,
        default_range: int = 5,
        feature_ranges=None,
    ):
        """Génère aléatoirement toutes les probas nécessaires, et le générateur d'aléatoire
        
        
        Keyword Arguments:
            nb_classes {int} -- [nombre de labels/classes] (default: {3})
            nb_features {int} -- [dimension de l'espace des données] (default: {4})
            use_softmax {bool} -- [utiliser softmax à la place d'argmax] (default: {False})
            seed {int} -- [graine pour le générateur d'aléatoire] (default: {None})
            default_range {int} -- [l'étalement par défaut des features si feature_ranges est None] (default: {5})
            feature_ranges {array-like} -- [les arguments de chaque range() qui définit les valeurs possibles des features] (default: {None})
        """

        self.nb_classes = nb_classes
        self.nb_features = nb_features
        self.use_softmax = use_softmax

        # NB: il vaut mieux utiliser default_rng maintenant, cf doc de numpy.random
        self.rng = default_rng(seed)

        self.prop = self.uniform_sum_one(self.nb_classes)

        if feature_ranges is not None:
            self.set_vals(feature_ranges)
        else:
            self.vals = [
                [i for i in range(default_range)] for _ in range(self.nb_features)
            ]

        self.generate_probas()

        # tous les points possibles
        # NB: tous les X possibles n'ont pas forcément P(X)!=0
        self.all_data_points = list(product(*self.vals))

    def set_vals(self, feature_ranges):
        """set les valeurs possibles pour chaque feature
        
        
        Arguments:
            feature_ranges {array-like} -- [Contient, pour chaque feature, 
            un itérable de la même forme que ceux passés à la fonction range() (cf sa documentation)]

        Note: met à disposition l'attribut vals
        """
        self.vals = [[i for i in range(*k_range)] for k_range in feature_ranges]

    def generate_probas(self):
        """Génère toutes les probabilités conditionnelles P(Xi=a|Y=k)
        
        Note: met à disposition l'attribut probas
        """

        self.probas = [
            [self.uniform_sum_one(len(self.vals[i])) for _ in range(self.nb_classes)]
            for i in range(self.nb_features)
        ]

    def uniform_sum_one(self, n: int):
        """Génère un vecteur de n probabilités uniformément réparties entre 0 et 1, de somme 1
        
        Arguments:
            n {int} -- [la longueur du vecteur à générer]
        """
        proba = [0.0, 1.0] + [self.rng.random() for i in range(n - 1)]
        proba.sort()
        prop = [proba[i + 1] - proba[i] for i in range(n)]
        return prop

    def __PX_inds__(self, X_inds):
        """Renvoie P(X)
        
        Doit être appelée avec les *indices* des valeurs des features de X et non X directement
        Cf. __PX__()
        
        Arguments:
            X_inds {array-like} -- [la liste des indices des valeurs prises par X dans les ranges des features]
        
        Returns:
            [float] -- [P(X)]
        """
        p = 0
        for k in range(self.nb_classes):  # pour chaque classe
            p_sach_k = self.prop[k]
            for i in range(self.nb_features):
                p_sach_k *= self.probas[i][k][X_inds[i]]
            p += p_sach_k
        return p

    def __PX__(self, X):
        """Calcule P(X)
        
        Arguments:
            X {array-like} -- le point à considérer
        
        Returns:
            [float] -- [P(X)]
        """
        return self.__PX_inds__(
            [self.vals[i].index(X[i]) for i in range(self.nb_features)]
        )

    def __bayes_layer__(self, X):
        """Couche bayésienne du classifieur
        
        Arguments:
            X {array-like} -- [le point à considérer]
        
        Returns:
            [list] -- [vecteur de taille nb_classes et dont l'élément i est P(Y=i|X)]
        """
        res = np.zeros(
            self.nb_classes
        )  # notre vecteur score en sortie, de dim le nb de classes
        p_x = self.__PX__(X)  # calcul de P(X)
        for k in range(self.nb_classes):
            p_inter = self.prop[k]  # p(Y)
            for i in range(self.nb_features):  # le nombre de features
                p_inter *= self.probas[i][k][
                    self.vals[i].index(X[i])
                ]  # p(Xi) sachant Y=k
            res[k] = p_inter / p_x
        return res

    def classify(self, X, display=True):
        """Classifie le point X dans une des features
        
        Arguments:
            X {array-like} -- [point de l'espace des features, de taille 1*nb_features]
        
        Keyword Arguments:
            display {bool} -- [si la classification devrait afficher un message ou non] (default: {True})
        
        Returns:
            [list] -- [vecteur de score de likelihood de chaque classe]
        """
        if self.use_softmax:
            return self.bayes_softmax(X, display)
        return self.bayes_argmax(X, display)

    def bayes_argmax(self, X, display=True):
        """Effectue la classification de X en utilisant argmax après la couche bayésienne
        
        Arguments:
            X {array-like} -- [le point à classifier]
        
        Keyword Arguments:
            display {bool} -- [si la classification devrait afficher un message ou non] (default: {True})
        
        Returns:
            [list] -- [l'encodage one-hot de la classe attribuée]
        """
        res = np.zeros(self.nb_classes)
        k = argmax(self.__bayes_layer__(X))
        res[k] = 1
        if display:
            print(f"Classe: {k}")
        return res  # one hot final

    def bayes_softmax(self, X, display=True):
        """Effectue la classification de X en utilisant softmax après la couche bayésienne
        
        Arguments:
            X {array-like} -- [le point à classifier]
        
        Keyword Arguments:
            display {bool} -- [si la classification devrait afficher un message ou non] (default: {True})
        
        Returns:
            [list] -- [vecteur post-softmax de score de chaque classe]
        """
        a = softmax(self.__bayes_layer__(X))  # array softmax
        if display:
            for k in range(3):
                print(f"Classe {k}. Confiance: {a[k]:.4f}")
        return a

    def __Rk__(self):
        """renvoie le vecteur des Rk pour k parcourant les classes
        """
        R = np.zeros(self.nb_classes)
        nb_points = len(self.all_data_points)
        for p in range(nb_points):
            X = self.all_data_points[p]
            if self.__PX__(X) != 0:
                if self.use_softmax:
                    R += self.bayes_softmax(X, False)
                else:
                    R += self.bayes_argmax(X, False)
        R /= nb_points
        return 1 - R

    def R(self):
        """renvoie le risque global du classifieur
        
        Returns:
            [R(C)] -- [risque global, dépend de tous les paramètres du classifieur]
        """
        return np.dot(self.prop, self.__Rk__())

    def plot_risk(self, npts: int = 300, keep_old_prop: bool = True):
        """Plot le risque pour npts valeurs différentes des probabilités a priori,
        pour argmax et softargmax
        
        Keyword Arguments:
            npts {int} -- [le nombre d'échantillons à utiliser] (default: {300})
            keep_old_prop {bool} -- [si l'on doit revenir aux proportions originales après le plotting] (default: {True})
        """
        if keep_old_prop:
            old_prop = self.prop
        old_fn = self.use_softmax

        if self.nb_classes == 2:
            self.__plot_2d__(npts)
        elif self.nb_classes == 3:
            self.__plot_3d__(npts)
        else:
            print("Le nombre de classes ne permet pas de plot le risque!")

        if keep_old_prop:
            self.prop = old_prop
        self.use_softmax = old_fn

    def __plot_2d__(self, npts):
        """Plot le risque pour npts valeurs différentes des 2 probabilités a priori,
        pour argmax et softargmax

        Note: pour une proba a priori donnée, l'autre se déduit, d'où le plot 2d
        
        Arguments:
            npts {int} -- [le nombre d'échantillons à utiliser]
        """
        risks_hard = []
        risks_soft = []
        X = []
        for _ in range(npts):
            self.prop = self.uniform_sum_one(self.nb_classes)
            X.append(self.prop[0])

            self.use_softmax = False
            risks_hard.append(self.R())

            self.use_softmax = True
            risks_soft.append(self.R())

        # argmax en rouge, softmax en vert
        plt.plot(X, risks_hard, "or", label="Avec Argmax")
        plt.plot(X, risks_soft, "og", label="Avec Softmax")

        plt.xlabel("pi1")
        plt.ylabel("risque")

        plt.title("Risque en fonction de la répartition dans les classes")
        plt.legend()
        plt.show()

    def __plot_3d__(self, npts):
        """Plot le risque pour npts valeurs différentes des 3 probabilités a priori,
        pour argmax et softargmax

        Note: pour deux probas a priori données, l'autre se déduit, d'où le plot 3d
        
        Arguments:
            npts {int} -- [le nombre d'échantillons à utiliser]
        """
        # fig = plt.figure()
        ax = plt.axes(projection="3d")

        risks_hard = []
        risks_soft = []
        X = []
        Y = []
        for _ in range(npts):
            self.prop = self.uniform_sum_one(self.nb_classes)
            X.append(self.prop[0])
            Y.append(self.prop[1])

            self.use_softmax = False
            risks_hard.append(self.R())

            self.use_softmax = True
            risks_soft.append(self.R())

        # argmax en rouge, softmax en vert
        ax.scatter3D(X, Y, risks_hard, c="r", label="Avec Argmax")
        ax.scatter3D(X, Y, risks_soft, c="g", label="Avec Softmax")

        ax.set_xlabel("pi1")
        ax.set_ylabel("pi2")
        ax.set_zlabel("Risque")

        plt.title("Risque en fonction de la répartition dans les classes")

        ax.legend()
        plt.show()


class BayesClassifierVect:
    """Cette classe implémente tout le processus d'un classifieur de bayes avec connaissance idéale
    des probabilités, sous forme vectorielle
    
    NOTE: ce classifieur n'est pas pensé pour pouvoir classifier des données aléatoires,
    car sa matrice de décision contient un nombre fixé de données.
    Ce n'est pas un problème si nb_pts n'est pas fixé (et que le classifieur a donc classifié tous les points possibles)
    """

    def __init__(
        self,
        nb_classes: int = 3,
        nb_features: int = 4,
        nb_pts: int = None,
        seed: int = None,
        prop_override=None,
        probas_override=None,
        default_range: int = 5,
        feature_ranges=None,
    ):
        """        
        Keyword Arguments:
            nb_classes {int} -- [nombre de labels/classes] (default: {3})
            nb_features {int} -- [dimension de l'espace des données] (default: {4})
            nb_pts{int} -- [nombre de points (attention à ne pas dépasser le nb de pts possibles) 
                            Quand None, on prend tous les points possibles] (default: {None})
            seed {int} -- [graine pour le générateur d'aléatoire] (default: {None})
            prop_override -- liste des probas a priori si on veut les définir manuellement
            probas_override -- array des probas conditionnelles si on veut les définir manuellement
            default_range {int} -- [l'étalement par défaut des features si feature_ranges est None] (default: {5})
            feature_ranges {array-like} -- [les arguments de chaque range() qui définit les valeurs possibles des features] (default: {None})
        """

        self.nb_classes = nb_classes
        self.nb_features = nb_features

        # NB: il vaut mieux utiliser default_rng maintenant, cf doc de numpy.random
        self.rng = default_rng(seed)

        if prop_override is not None:
            self.prop = prop_override
        else:
            self.prop = self.uniform_sum_one(self.nb_classes)

        if feature_ranges is not None:
            self.set_vals(feature_ranges)
        else:
            self.vals = [
                [i for i in range(default_range)] for _ in range(self.nb_features)
            ]

        if probas_override is not None:
            self.probas = probas_override
        else:
            self.generate_probas()

        # tous les points
        all_possible_points = list(product(*self.vals))
        if nb_pts is not None:
            self.nb_pts = nb_pts
            self.all_data_points = all_possible_points[:nb_pts]
        else:
            self.nb_pts = len(all_possible_points)
            self.all_data_points = all_possible_points

        self.update_A_D()

    def set_vals(self, feature_ranges):
        """set les valeurs possibles pour chaque feature
        
        
        Arguments:
            feature_ranges {array-like} -- [Contient, pour chaque feature, 
            un itérable de la même forme que ceux passés à la fonction range() (cf sa documentation)]

        Note: met à disposition l'attribut vals
        """
        self.vals = [[i for i in range(*k_range)] for k_range in feature_ranges]

    def generate_probas(self):
        """Génère toutes les probabilités conditionnelles P(Xi=a|Y=k)
        
        Note: met à disposition l'attribut probas
        """

        self.probas = [
            [self.uniform_sum_one(len(self.vals[i])) for _ in range(self.nb_classes)]
            for i in range(self.nb_features)
        ]

    def uniform_sum_one(self, n: int):
        """Génère un vecteur de n probabilités uniformément réparties entre 0 et 1, de somme 1
        
        Arguments:
            n {int} -- [la longueur du vecteur à générer]
        """
        proba = [0.0, 1.0] + [self.rng.random() for i in range(n - 1)]
        proba.sort()
        prop = [proba[i + 1] - proba[i] for i in range(n)]
        return prop

    def update_A_D(self):
        """Fonction qui calcule les matrices de probabilités et de décision,
        selon les expressions donnée dans le rapport
        
        Note: doit être appelée après chaque changement des probabilités à priori
        """
        self.A = self.get_A()
        self.D = self.get_D()

    def get_A(self):
        """Matrice des probabilités (cf rapport)
        
        Returns:
            [array] -- [matrice de taille (nb_classes*nb_pts) et dont l'élément ij est P(Y=i|Xi)]
        """
        mat_probas = np.zeros((self.nb_classes, self.nb_pts))

        for j, X in enumerate(self.all_data_points):
            mat_probas[:, j] = self.__proba_cond__(X)

        return np.reshape(self.prop, (self.nb_classes, 1)) * mat_probas

    def __proba_cond__(self, X):

        p_vect = np.zeros((1, self.nb_classes))
        for k in range(self.nb_classes):
            p_cond = 1
            for i in range(self.nb_features):  # le nombre de features
                p_cond *= self.probas[i][k][
                    self.vals[i].index(X[i])
                ]  # p(Xi) sachant Y=k
            p_vect[0, k] = p_cond
        return p_vect

    def get_D(self):
        return softmax(self.A, axis=0)

    def classify(self, X, display=True):
        """Classifie le point X dans une des features
        
        Arguments:
            X {array-like} -- [point de l'espace des features, de taille 1*nb_features]
        
        Keyword Arguments:
            display {bool} -- [si la classification devrait afficher un message ou non] (default: {True})
        
        Returns:
            [list] -- [vecteur de score de likelihood de chaque classe]
        """
        j = self.all_data_points.index(X)
        res = self.D[:, j]
        if display:
            for k in range(self.nb_features):
                print(f"Classe {k}. Confiance: {res[k]:.4f}")
        return res

    def __nkro__(self, i, j):
        """equivalent a 1 - kronecker(i, j)"""
        if i == j:
            return 0
        else:
            return 1

    def __somme_terme_2__(self, i, j):
        """second terme dans la somme selon j dans le gradient, séparé ici pour des raisons de clarté"""
        res = 0
        for k in range(self.nb_classes):
            if k != i:
                res += self.__nkro__(k, j) * (-self.D[k, j]) * self.D[i, j]
        return res

    def gradient(self, pi):
        """renvoie grad(R(D, prior))(pi). Simple implémentation de la formule démontrée dans le rapport"""

        res = np.zeros(self.nb_classes)

        old_prop = self.prop  # on garde les prop initiaux
        self.prop = pi
        self.update_A_D()  # re-calcul de A et de D avec le nouveau pi

        for i in range(self.nb_classes):
            for j in range(self.nb_pts):
                Sij = self.D[i, j]  # softmax pour la classe i, donnée j
                res[i] += self.__nkro__(i, j) * (Sij + self.prop[i] * Sij * (1 - Sij))
                res[i] += self.__somme_terme_2__(i, j)

        # on laisse les attributs comme on les avait trouvés
        self.prop = old_prop
        self.update_A_D()

        return res


class GradientProjete:
    """Cette classe implément l'algorithme du gradient projeté,
    pour trouver un maximum d'une fonction dont on connait le gradient
    
    Note: la convergence vers le maximum réel n'est nullement garantie par cette classe
    """

    def __init__(self, gradient, X_init, n_step: int = 100, rate: float = 0.01):
        """initialise l'algorithme
        
        [description]
        
        Arguments:
            gradient {function} -- [gradient de la fonction dont on cherche le maximum]
            X_init {array-like} -- [vecteur initial]
        
        Keyword Arguments:
            n_step {int} -- [nombre d'étapes à effcteuer] (default: {100})
            rate {float} -- [learning rate] (default: {.01})
        """

        self.gradient = gradient
        self.X_init = X_init
        self.n_step = n_step
        self.lr = rate

    def find_maximum(self, log_interval=10):
        """Effectue les étapes de l'algo du gradient projeté
        
        Keyword Arguments:
            log_interval {number} -- [la période d'impression vers stdout du X actuel] (default: {10})
        """

        n = 0
        X = self.X_init

        while n <= self.n_step:

            if n % log_interval == 0:
                print(f"Etape: {n} | {X}")

            X_prime = -self.gradient(X)
            X = X + self.lr * X_prime
            X = self.__proj_simpl_vector__(X)
            n += 1

        return X

    def __proj_simpl_vector__(self, y):
        """ Projete un vecteur sur le simplex [0,1]

        Arguments : 
        y {array-like} -- {vecteur à projeter }

        Returns : 

        [array-like] -- [ vecteur projeté sur le simplex [0,1] ]
        """
        z = np.cumsum(sorted(y, reverse=True)) - np.ones(len(y))
        a = list(range(1, len(y) + 1))
        maxi = max(np.divide(z, a))
        end = np.maximum(y - maxi * np.ones(len(y)), 0)

        return end

