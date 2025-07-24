# Réseau de Neurones pour la Reconnaissance de Chiffres MNIST

Ce projet présente l'implémentation d'un réseau de neurones simple à partir de zéro en utilisant uniquement la bibliothèque **NumPy**. L'objectif est de reconnaître des chiffres manuscrits provenant du célèbre dataset MNIST.

Ce travail a été réalisé dans le but de comprendre en profondeur les mécanismes internes d'un réseau de neurones, notamment la **propagation avant (forward propagation)**, la **rétropropagation (backpropagation)** et la **descente de gradient (gradient descent)**.

## 🚀 Fonctionnalités

- **Aucune bibliothèque de haut niveau** : Le réseau est construit sans TensorFlow, PyTorch ou Scikit-learn.
- **Architecture simple** : Un réseau avec une couche d'entrée, une couche cachée et une couche de sortie.
- **Fonctions d'activation** : Utilisation de **ReLU** pour la couche cachée et **Softmax** pour la couche de sortie.
- **Préparation des données** : Un notebook est inclus pour charger le dataset MNIST original et le convertir au format CSV.
- **Entraînement et Évaluation** : Le modèle est entraîné et sa performance est évaluée sur un ensemble de validation distinct.

## 🛠️ Comment l'utiliser

### Prérequis

- Python 3.x
- Les bibliothèques listées dans `requirements.txt`

### Installation

1.  Clonez ce dépôt :
    ```bash
    git clone https://github.com/Linwe-e/Neural-Network-MNIST-Numpy-.git
    ```
2.  Naviguez dans le répertoire du projet :
    ```bash
    cd Neural-Network-MNIST-Numpy-
    ```
3.  Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

### Exécution

1.  **Préparation des données** :
    - Ouvrez et exécutez le notebook `ReadMNISTDataset.ipynb`. Cela générera les fichiers `mnist_train_full.csv` et `mnist_test_full.csv`.

2.  **Entraînement du modèle** :
    - Ouvrez et exécutez le notebook `MNISTDigitRecognition.ipynb`. Vous pouvez y ajuster les hyperparamètres (nombre de neurones, taux d'apprentissage, nombre d'itérations) et observer l'entraînement en temps réel.

## 🧠 Concepts implémentés

- **Initialisation des poids et des biais**
- **Forward Propagation**
- **Calcul de la fonction de coût (Cross-Entropy)**
- **Backpropagation**
- **Mise à jour des paramètres via la descente de gradient**

## 📊 Expérimentations et Résultats

Plusieurs configurations d'hyperparamètres ont été testées pour optimiser la performance du modèle. Voici un résumé des expérimentations :

| Couche Cachée | Taux d'apprentissage | Itérations | Précision (Validation) |
| :-----------: | :------------------: | :--------: | :--------------------: |
| 10 neurones   | 0.1                  | 500        | ~85%                   |
| 10 neurones   | 0.01                 | 700        | ~73% (sous-apprentissage) |
| **28 neurones**   | **0.05**             | **2000**   | **~91.1%**             |

La meilleure configuration obtenue atteint **91.10%** de précision sur l'ensemble de validation.

---

Ce projet est une excellente base pour quiconque souhaite apprendre les fondements des réseaux de neurones de manière pratique.
