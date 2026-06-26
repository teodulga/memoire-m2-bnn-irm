# Explicabilité et fiabilité des réseaux de neurones pour la détection de tumeur cérébrale

![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8-orange)
![torchbnn](https://img.shields.io/badge/torchbnn-BNN-purple)

Mémoire Master 2 Intelligence Artificielle - École Hexagone 2025-2026  
Téodul Guiraud-Allègre

## Résumé

L'imagerie des tumeurs cérébrales repose presque exclusivement sur l'IRM, et les modèles de deep learning atteignent aujourd'hui des performances comparables à celles d'un expert. Cependant, on ne sait pas vraiment sur quoi reposent leurs décisions ni quelle confiance leur accorder, ce qui pose problème en contexte clinique. Ce mémoire cherche à améliorer l'explicabilité et la fiabilité d'un modèle de classification d'IRM cérébrales à 4 classes (gliome, méningiome, tumeur pituitaire, absence de tumeur). J'ai d'abord entraîné un CNN de référence déterministe (ResNet18), puis construit une architecture bayésienne qui combine ce même ResNet18 avec une tête bayésienne (torchbnn, 50 passes Monte-Carlo). J'ai aussi testé une variante multimodale intégrant une estimation de l'âge cérébral, obtenue par un EfficientNet-B0 entraîné sur le dataset IXI. J'ai comparé ce BNN à une variante MC Dropout et validé les trois modèles sur le dataset externe Br35H.

J'ai constaté que 61.6 % des images de test étaient présentes dans le train. J'ai donc reconstruit un split propre (4 606 images d'entraînement, 1 149 de test, zéro doublon). Le passage au bayésien préserve les performances (AUC 0.995 contre 0.998 pour le CNN) tout en apportant une mesure d'incertitude par prédiction : les prédictions incorrectes sont 4.83 fois plus incertaines que les correctes. Ce BNN est mieux calibré que le CNN (ECE 0.0085 contre 0.0160). La variante multimodale dégrade la calibration (ECE 0.0276) et n'apporte pas de gain de performance.

En distribution, trier les prédictions par incertitude croissante permet de couvrir 87.8 % des erreurs en ne renvoyant que 20 % des images à un radiologue. torchbnn produit une incertitude en distribution 20 fois plus élevée que MC Dropout (σ médian 0.0217 contre 0.0010). MC Dropout est presque déterministe et ne permet pas de définir un seuil de triage clinique.

Ces chiffres correspondent à la seed de référence (42), qui s'est avérée la plus favorable. Sur les cinq seeds, le ratio σ<sub>incorrect</sub> / σ<sub>correct</sub> reste supérieur à 2 (3.69 en moyenne) et le triage couvre toujours la majorité des erreurs (83.2 % en moyenne).

Au-delà de l'incertitude, les cartes Grad-CAM montrent sur quelles zones le modèle s'appuie, et leur variance sur 50 passes permet d'explorer où son attention varie le plus.

**Ce travail montre qu'on peut rendre un modèle de classification IRM plus transparent et plus fiable sans sacrifier ses performances.**

**Mots-clés :** classification IRM cérébrale, réseau bayésien, incertitude, Grad-CAM, calibration, explicabilité, MC Dropout, validation externe, qualité des données

## Abstract

Brain tumor imaging relies almost exclusively on MRI, and deep learning models now reach performance comparable to that of an expert. Yet it usually remains unclear what their decisions rely on and how much confidence to place in them, which is a problem in a clinical setting. This thesis aims to improve the explainability and reliability of a four-class brain MRI classification model (glioma, meningioma, pituitary tumor, no tumor). I first trained a deterministic baseline CNN (ResNet18), then built a Bayesian architecture combining this same ResNet18 with a Bayesian head (torchbnn, 50 Monte-Carlo passes). I also tested a multimodal variant incorporating a brain-age estimate, produced by an EfficientNet-B0 trained on the IXI dataset. I compared this BNN to an MC Dropout variant and validated all three models on the external Br35H dataset.

I found that 61.6% of test images were present in the training set. I therefore rebuilt a clean split (4,606 training images, 1,149 test images, zero duplicates). Moving to a Bayesian model preserves performance (AUC 0.995 versus 0.998 for the CNN) while providing a per-prediction uncertainty measure: incorrect predictions are 4.83 times more uncertain than correct ones. This BNN is better calibrated than the CNN (ECE 0.0085 versus 0.0160). The multimodal variant degrades calibration (ECE 0.0276) and does not improve performance.

In-distribution, sorting predictions by uncertainty covers 87.8% of errors by sending only 20% of images to a radiologist. torchbnn produces an in-distribution uncertainty 20 times higher than MC Dropout (σ median 0.0217 versus 0.0010). MC Dropout is nearly deterministic and does not allow a clinical triage threshold to be defined.

These figures correspond to the reference seed (42), which turned out to be the most favorable. Across the five seeds, the σ<sub>incorrect</sub> / σ<sub>correct</sub> ratio stays above 2 (3.69 on average) and triage still covers the majority of errors (83.2% on average).

Beyond uncertainty, Grad-CAM maps show which regions the model relies on, and their variance over 50 passes allows exploring where its attention varies the most.

**This work shows that a brain MRI classification model can be made more transparent and more reliable without sacrificing performance.**

**Keywords:** brain MRI classification, Bayesian neural network, uncertainty, Grad-CAM, calibration, explainability, MC Dropout, external validation, data quality

## Ordre d'exécution

0. `00_clean_split.ipynb` - Déduplication et split propre
1. `01_cnn_baseline.ipynb` - CNN baseline ResNet18
2. `02_brain_age_prediction.ipynb` - Prédicteur d'âge cérébral BrainAgeNet
3. `03_bnn_multimodal.ipynb` - BNN multimodal torchbnn
4. `04_clinical_analysis.ipynb` - Analyse incertitude et calibration
5. `05_gradcam.ipynb` - Grad-CAM et explicabilité visuelle
6. `06_bnn_mc_dropout.ipynb` - Variante MC Dropout
7. `07_br35h.ipynb` - Validation externe Br35H
8. `08_seed_robustness.ipynb` - Robustesse inter-seed
9. `09_extra.ipynb` - Analyses complémentaires

## Installation

```bash
pip install -r requirements.txt
```

## Datasets

- Mendeley BT-MRI : https://data.mendeley.com/datasets/zwr4ntf94j/4
- IXI : https://brain-development.org/ixi-dataset/
- Br35H : https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection

## Environnement

MacBook Pro Apple M5, Python 3.9