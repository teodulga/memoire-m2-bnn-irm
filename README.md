# Explicabilité et fiabilité des réseaux de neurones pour la détection de tumeur cérébrale

![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8-orange)
![torchbnn](https://img.shields.io/badge/torchbnn-BNN-purple)

Mémoire Master 2 Intelligence Artificielle - École Hexagone 2025-2026  
Téodul Guiraud

## Résumé

Les tumeurs cérébrales se diagnostiquent quasi-exclusivement par IRM, et les modèles de deep learning atteignent aujourd'hui des performances comparables à celles d'un expert. Cependant, on ne sait pas vraiment sur quoi reposent leurs décisions ni quelle confiance leur accorder, ce qui pose problème en contexte clinique. Ce mémoire cherche à améliorer l'explicabilité et la fiabilité d'un modèle de classification d'IRM cérébrales à 4 classes (gliome, méningiome, tumeur pituitaire, absence de tumeur). J'ai d'abord entraîné un CNN de référence déterministe (ResNet18), puis construit une architecture bayésienne multimodale qui combine ce même ResNet18 avec une tête bayésienne (torchbnn, 50 passes Monte-Carlo) et une estimation de l'âge cérébral obtenue par un EfficientNet-B0 entraîné sur le dataset IXI. J'ai comparé cette approche à une variante MC Dropout et validé les modèles sur le dataset externe Br35H. Le passage au bayésien préserve les performances (AUC 0.995 contre 0.997 pour le CNN) tout en apportant une mesure d'incertitude par prédiction : **les prédictions incorrectes sont en moyenne 7.41 fois plus incertaines que les correctes**, et le modèle reste bien calibré (ECE 0.0184). Les cartes Grad-CAM calculées sur 50 passes ajoutent une carte de variance spatiale de l'attention. Sur Br35H, torchbnn détecte le changement de distribution plus nettement (ratio σ 2.55, p=2.22e-25) et de façon stable, alors que la sensibilité de MC Dropout varie selon le taux de dropout choisi. En ne gardant que les prédictions les plus confiantes, l'accuracy passe de 0.778 à 0.888. Ce travail montre **qu'on peut rendre un modèle de classification IRM plus transparent et plus fiable sans sacrifier ses performances**.

**Mots-clés :** classification IRM cérébrale, réseau bayésien, incertitude, Grad-CAM, fusion multimodale, calibration, explicabilité, MC Dropout, validation externe

## Abstract

Brain tumors are diagnosed almost exclusively through MRI, and deep learning models now reach performance comparable to that of an expert. Yet it usually remains unclear what their decisions rely on and how much confidence to place in them, which is a problem in a clinical setting. This thesis aims to improve the explainability and reliability of a four-class brain MRI classification model (glioma, meningioma, pituitary tumor, no tumor). I first trained a deterministic baseline CNN (ResNet18), then built a multimodal Bayesian architecture combining this same ResNet18 with a Bayesian head (torchbnn, 50 Monte-Carlo passes) and a brain-age estimate produced by an EfficientNet-B0 trained on the IXI dataset. I compared this approach to an MC Dropout variant and validated the models on the external Br35H dataset. Moving to a Bayesian model preserves performance (AUC 0.995 versus 0.997 for the CNN) while providing a per-prediction uncertainty measure: **incorrect predictions are on average 7.41 times more uncertain than correct ones**, and the model stays well calibrated (ECE 0.0184). Grad-CAM maps computed over 50 passes add a spatial variance map of the model's attention. On Br35H, torchbnn detects the distribution shift more clearly (σ ratio 2.55, p=2.22e-25) and consistently, while MC Dropout's sensitivity varies with the chosen dropout rate. Keeping only the most confident predictions raises accuracy from 0.778 to 0.888. This work shows that **a brain MRI classification model can be made more transparent and more reliable without sacrificing performance**.

**Keywords:** brain MRI classification, Bayesian neural network, uncertainty, Grad-CAM, multimodal fusion, calibration, explainability, MC Dropout, external validation

## Ordre d'exécution

1. `01_cnn_baseline.ipynb` - CNN baseline ResNet18
2. `02_brain_age_prediction.ipynb` - Prédicteur d'âge cérébral BrainAgeNet
3. `03_bnn_multimodal.ipynb` - BNN multimodal torchbnn
4. `04_clinical_analysis.ipynb` - Analyse incertitude et calibration
5. `05_gradcam.ipynb` - Grad-CAM et explicabilité visuelle
6. `06_bnn_mc_dropout.ipynb` - Variante MC Dropout
7. `07_br35h.ipynb` - Validation externe Br35H
8. `08_extra.ipynb` - Analyses complémentaires

## Installation

```bash
pip install -r requirements.txt
```

## Datasets

- Epic & CSCR Hospital Brain Tumor MRI : https://data.mendeley.com/datasets/zwr4ntf94j/4
- IXI : https://brain-development.org/ixi-dataset/
- Br35H : https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection

## Environnement

MacBook Pro Apple M5, Python 3.9