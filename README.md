# Exercice de Programmation
## Classification d'Images de Chiens, Chats et Oiseaux avec un CNN

### 1. Préparation des données
- Téléchargement des images brutes depuis https://www.kaggle.com/datasets/swaroopkml/cifar10-pngs-in-folders?resource=download
- Dossier /train :
  - 5000 photos / catégorie
  - utilisé pour l'entrainement et la validation (80/20) 
- Dossier /test :
  - 1000 photos / catégorie
  - utilisé pour les tests
- Extraction des sous-dossiers nécessaires : /dog, /cat, /bird

### 2. Prétraitement des Images
- Utilisation de ImageDataGenerator() pour 
  - la normalisation des images
  - la séparation entrainement / validation
- TODO: Si nécessaire augmenter virtuellement le nombre d'images par des transformation (flip, roration)

### 3. Création du Modèle CNN
- Fichier : 3.CNN-Projet1-model-tf.py
- Modèle Sequential Keras
- Compilation avec 
  - perte par cross-entropie catégorielle
  - optimizer = Adam
  - metrics = Accuracy
- Entrainement avec Early Stop:
  - patience = 2 pour attendre 2 epoch avant de stopper
  - restore_best_weights pour revenir au meilleur score
  - Résultats V1: Best Epoch = 7, Accuracy = 2e-04 🥹 
- Visualisation graphique / epoch
  - perte à l'entraînement
  - perte au test
- Sauvegarde du modèle

### 4. Visualisation des Couches Convolutionnelles
- Dossier /v1
- maps-- + nom de la couche
- Exemple :
  - ![maps--model_cnn_cifar10-conv2d.png](tf_models%2Fv1%2Fmaps--model_cnn_cifar10-conv2d.png)
- features_map-- + nom de la couche
- Exemple
  - ![feature_maps--model_cnn_cifar10-conv2d.png](tf_models%2Fv1%2Ffeature_maps--model_cnn_cifar10-conv2d.png)

### 5. Évaluation et Test
- Fichier : 3.CNN-Projet1-test-tf.py
- Résultats Version 1 :
  - Evaluation mauelle : 4 passés / 8 tests 🥹
  - Evaluation automatique avec jeu de test : 
    - Précision = 11% 😭

### 6. Amélioration
- V2.1 :
  - Ajout d'images virtuellement par augmentation :
    - RandomFlip("horizontal_and_vertical")
    - RandomRotation(0.2)
  - Augmentation Epoque : 50 (avec toujours le early stop)
  - Résultats Version 2.1 : 
    - 68% 🥹
- V2.2 :
  - Suppression des augmentations
  - Augmentation des couches de conv
  - Augmentation des couches de neurones
  - Ajout de couches de DropOut (après Pooling) pour mitiger l'apprentissage
- V2.3 :
  - Changement de l'optimizer par SGD
