# Rapport d'Évaluation des Modèles de Sentiment

Ce rapport compare les performances de VADER (avec lexique crypto) et FinBERT sur un ensemble de données de validation étiquetées manuellement.

## Tableau Comparatif des Métriques
| Modèle   | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-score (Weighted) |
|----------|----------|----------------------|-------------------|---------------------|
| VADER | 0.4750 | 0.4832 | 0.4750 | 0.4764 |
| FinBERT | 0.5167 | 0.6361 | 0.5167 | 0.4721 |

## Matrices de Confusion
### VADER
![Matrice de Confusion VADER](confusion_matrix_VADER.png)

### FinBERT
![Matrice de Confusion FinBERT](confusion_matrix_FinBERT.png)

## Analyse des Erreurs
Une analyse plus approfondie des cas où les modèles se trompent peut être effectuée pour identifier les faiblesses (par exemple, sarcasme, contexte complexe).
