<a name="version-francaise"></a>
# Réseau de Neurones pour la Reconnaissance de Chiffres MNIST

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Only-orange?style=for-the-badge&logo=numpy&logoColor=white)
![MNIST](https://img.shields.io/badge/Dataset-MNIST-green?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Accuracy-91.1%25-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> 🇫🇷 **Version française** | 🇺🇸 **[English version below](#english-version)** (voir plus bas)

Ce projet présente l'implémentation d'un réseau de neurones simple à partir de zéro en utilisant uniquement la bibliothèque **NumPy**. L'objectif est de reconnaître des chiffres manuscrits provenant du célèbre dataset MNIST.

Ce travail a été réalisé dans le but de comprendre en profondeur les mécanismes internes d'un réseau de neurones, notamment la **propagation avant (forward propagation)**, la **rétropropagation (backpropagation)** et la **descente de gradient (gradient descent)**.

## 🚀 Fonctionnalités

![No Dependencies](https://img.shields.io/badge/Dependencies-Minimal-lightblue?style=flat-square)
![From Scratch](https://img.shields.io/badge/Implementation-From%20Scratch-purple?style=flat-square)
![Neural Network](https://img.shields.io/badge/Architecture-Simple%20NN-red?style=flat-square)

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

![Best Config](https://img.shields.io/badge/Best%20Config-28%20neurons-success?style=flat-square)
![Learning Rate](https://img.shields.io/badge/Learning%20Rate-0.05-informational?style=flat-square)
![Iterations](https://img.shields.io/badge/Iterations-2000-blueviolet?style=flat-square)

Plusieurs configurations d'hyperparamètres ont été testées pour optimiser la performance du modèle. Voici un résumé des expérimentations :

| Couche Cachée | Taux d'apprentissage | Itérations | Précision (Validation) |
| :-----------: | :------------------: | :--------: | :--------------------: |
| 10 neurones   | 0.1                  | 500        | ~85%                   |
| 10 neurones   | 0.01                 | 700        | ~73% (sous-apprentissage) |
| **28 neurones**   | **0.05**             | **2000**   | **~91.1%**             |

La meilleure configuration obtenue atteint **91.10%** de précision sur l'ensemble de validation.

## 📚 Ressources et Références

Ce projet s'inspire et utilise les ressources suivantes :

- **📄 Article Medium** : [Building a Simple Neural Network from Scratch for MNIST Digit Recognition](https://medium.com/@ombaval/building-a-simple-neural-network-from-scratch-for-mnist-digit-recognition-without-using-7005a7733418)
- **🗂️ Dataset MNIST** : [MNIST Original Dataset sur Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
- **📓 Notebook de lecture MNIST** : [Read MNIST Dataset Notebook](https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook)

---

Ce projet est une excellente base pour quiconque souhaite apprendre les fondements des réseaux de neurones de manière pratique.

---

<a name="english-version"></a>
# 🇺🇸 English Version

# Neural Network for MNIST Digit Recognition

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Only-orange?style=for-the-badge&logo=numpy&logoColor=white)
![MNIST](https://img.shields.io/badge/Dataset-MNIST-green?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Accuracy-91.1%25-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> 🇺🇸 **English version** | 🇫🇷 **[Version française ci-dessus](#version-francaise)** (see above)

This project presents the implementation of a simple neural network from scratch using only the **NumPy** library. The goal is to recognize handwritten digits from the famous MNIST dataset.

This work was carried out to understand in depth the internal mechanisms of a neural network, particularly **forward propagation**, **backpropagation** and **gradient descent**.

## 🚀 Features

![No Dependencies](https://img.shields.io/badge/Dependencies-Minimal-lightblue?style=flat-square)
![From Scratch](https://img.shields.io/badge/Implementation-From%20Scratch-purple?style=flat-square)
![Neural Network](https://img.shields.io/badge/Architecture-Simple%20NN-red?style=flat-square)

- **No high-level libraries**: The network is built without TensorFlow, PyTorch or Scikit-learn.
- **Simple architecture**: A network with an input layer, a hidden layer and an output layer.
- **Activation functions**: Use of **ReLU** for the hidden layer and **Softmax** for the output layer.
- **Data preparation**: A notebook is included to load the original MNIST dataset and convert it to CSV format.
- **Training and Evaluation**: The model is trained and its performance is evaluated on a separate validation set.

## 🛠️ How to use

### Prerequisites

- Python 3.x
- The libraries listed in `requirements.txt`

### Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/Linwe-e/Neural-Network-MNIST-Numpy-.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd Neural-Network-MNIST-Numpy-
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Execution

1.  **Data preparation**:
    - Open and run the `ReadMNISTDataset.ipynb` notebook. This will generate the `mnist_train_full.csv` and `mnist_test_full.csv` files.

2.  **Model training**:
    - Open and run the `MNISTDigitRecognition.ipynb` notebook. You can adjust the hyperparameters (number of neurons, learning rate, number of iterations) and observe the training in real time.

## 🧠 Implemented concepts

- **Weight and bias initialization**
- **Forward Propagation**
- **Cost function calculation (Cross-Entropy)**
- **Backpropagation**
- **Parameter update via gradient descent**

## 📊 Experiments and Results

![Best Config](https://img.shields.io/badge/Best%20Config-28%20neurons-success?style=flat-square)
![Learning Rate](https://img.shields.io/badge/Learning%20Rate-0.05-informational?style=flat-square)
![Iterations](https://img.shields.io/badge/Iterations-2000-blueviolet?style=flat-square)

Several hyperparameter configurations have been tested to optimize model performance. Here is a summary of the experiments:

| Hidden Layer | Learning Rate | Iterations | Accuracy (Validation) |
| :----------: | :-----------: | :--------: | :-------------------: |
| 10 neurons   | 0.1           | 500        | ~85%                  |
| 10 neurons   | 0.01          | 700        | ~73% (underfitting)   |
| **28 neurons**   | **0.05**      | **2000**   | **~91.1%**           |

The best configuration achieved **91.10%** accuracy on the validation set.

## 📚 Resources and References

This project is inspired by and uses the following resources:

- **📄 Medium Article**: [Building a Simple Neural Network from Scratch for MNIST Digit Recognition](https://medium.com/@ombaval/building-a-simple-neural-network-from-scratch-for-mnist-digit-recognition-without-using-7005a7733418)
- **🗂️ MNIST Dataset**: [MNIST Original Dataset on Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
- **📓 MNIST Reading Notebook**: [Read MNIST Dataset Notebook](https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook)

---

This project is an excellent foundation for anyone who wants to learn the fundamentals of neural networks in a practical way.
