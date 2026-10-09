# Parent-Child Height Prediction : Régression Linéaire en Machine Learning

Ce projet met en œuvre les fondements de l'analyse prédictive à travers l'étude de la relation biologique et biométrique entre la taille des parents et celle de leurs enfants. L'objectif est de concevoir, d'entraîner et d'évaluer des modèles de régression capables de généraliser sur de nouvelles données familières.

---

## 📑 Table des Matières
1. [Aperçu Théorique](#-aperçu-théorique)
2. [Exploration et Prétraitement des Données (Data Cleaning)](#-exploration-et-prétraitement-des-données-data-cleaning)
3. [Analyse Visuelle (EDA)](#-analyse-visuelle-eda)
4. [Modélisation par Régression Linéaire Simple](#-modélisation-par-régression-linéaire-simple)
5. [Extension : Régression Linéaire Multiple & Feature Engineering](#-extension--régression-linéaire-multiple--feature-engineering)
6. [Pipeline de Prédiction](#-pipeline-de-prédiction)
7. [Technologies Utilisées](#-technologies-utilisées)

---

## 🧠 Aperçu Théorique

Le projet s'appuie sur le modèle mathématique de la régression linéaire simple :
\[y = \beta_0 + \beta_1 x + \epsilon\]

*   **\(\beta_0\) (Intercept)** : L'ordonnée à l'origine, représentant la valeur moyenne théorique de la cible \(Y\) lorsque l'attribut \(X\) est nul.
*   **\(\beta_1\) (Coefficient/Pente)** : L'impact marginal de la variable indépendante sur la variable dépendante.
*   **\(\epsilon\) (Erreur aléatoire)** : Le terme résiduel modélisant les variations non expliquées, supposé suivre une loi normale centrée.

Pour estimer ces coefficients, la **Méthode des Moindres Carrés (OLS - Ordinary Least Squares)** est appliquée afin de minimiser la somme des carrés des résidus entre les observations réelles et les prédictions du modèle.

---

## 🧹 Exploration et Prétraitement des Données (Data Cleaning)

Une attention particulière a été accordée à la qualité des données avant l'apprentissage du modèle :
*   **Analyse des valeurs manquantes** : Identification de valeurs non numériques (`NaN`) au sein du dataset original (ex: ligne 1 de la colonne `Height`).
*   **Nettoyage (Dropna)** : Élimination des lignes incomplètes à l'aide de Pandas (`data1.dropna()`). La régression linéaire classique étant sensible aux données manquantes, cette étape sécurise la stabilité des calculs mathématiques.

---

## 📊 Analyse Visuelle (EDA)

Un nuage de points (*Scatter Plot*) a été généré avec `matplotlib.pyplot` pour explorer la structure des données :
1.  **Tendance Linéaire Positive** : Visualisation claire d'une trajectoire ascendante, confirmant la pertinence d'une modélisation linéaire.
2.  **Forte Corrélation** : Les points se regroupent de manière dense le long d'une droite imaginaire.
3.  **Variabilité Résiduelle** : Dispersion naturelle des points représentant la variance non captable par une seule variable (justifiant l'extension vers un modèle multiple).

---

## 📉 Modélisation par Régression Linéaire Simple

Entraîné sur la variable explicative `Midpart` (taille moyenne des parents) pour prédire `Height` (taille de l'enfant), le modèle Scikit-Learn a convergé vers les résultats suivants :
*   **Équation apprise** : \(y = -0.538 + 1.008 \times x\)
*   **Interprétation du coefficient (\(\beta_1 \approx 1.008\))** : Pour chaque augmentation de 1 cm de la taille moyenne des parents, la taille prédite de l'enfant augmente en moyenne de 1.008 cm.

---

## 🚀 Extension : Régression Linéaire Multiple & Feature Engineering

Pour dépasser les limites de la régression simple, un second pipeline intègre des fonctionnalités plus riches (`Father`, `Mother`, `Gender`) :

### 1. Encodage des Variables Catégorielles (*Label Encoding*)
La colonne `Gender` contenant du texte (`M`, `F`), l'outil `LabelEncoder` de Scikit-Learn a été déployé pour transformer ces catégories nominales en valeurs numériques discrètes (\(0\) pour Féminin, \(1\) pour Masculin), rendant la feature compatible avec l'algorithme.

### 2. Entraînement de la Régression Multiple
L'équation finale obtenue est :
\[y = -0.798 + 0.518 \times \text{Father} + 0.491 \times \text{Mother} + 0.278 \times \text{Gender}\]

*   **Impact du Sexe** : À caractéristiques parentales égales, un enfant de sexe masculin (\(1\)) présente une taille supérieure de 0.278 cm par rapport à un enfant de sexe féminine (\(0\)).
*   **Équilibre Biologique** : Les coefficients du père (\(0.518\)) et de la mère (\(0.491\)) s'équilibrent de manière cohérente avec la génétique.

---

## 🔮 Pipeline de Prédiction

En appliquant la méthode `.predict()`, le modèle démontre sa capacité de **généralisation** en calculant des estimations sur des jeux de données d'inférence non vus pendant la phase d'apprentissage (`prediction1.csv` et `prediction2.csv`).

---

## 🛠️ Technologies Utilisées
*   **Langage** : Python 3
*   **Traitement de Données** : Pandas, NumPy
*   **Visualisation** : Matplotlib
*   **Machine Learning** : Scikit-Learn (`LinearRegression`, `LabelEncoder`)
