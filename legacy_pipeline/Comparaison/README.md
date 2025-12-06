# Rapport d'�valuation des Mod�les de Sentiment

Ce rapport compare les performances de VADER, FinBERT et RoBERTa sur un ensemble de donn�es de validation �tiquet�es manuellement.

## Tableau Comparatif des M�triques
| Mod�le   | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-score (Weighted) |
|----------|----------|----------------------|-------------------|---------------------|
| VADER | 0.4750 | 0.4832 | 0.4750 | 0.4764 |
| FINBERT | 0.5167 | 0.6361 | 0.5167 | 0.4721 |
| ROBERTA | 0.5500 | 0.5954 | 0.5500 | 0.5355 |

## Matrices de Confusion
### VADER
![Matrice de Confusion VADER](confusion_matrix_VADER.png)

### FINBERT
![Matrice de Confusion FINBERT](confusion_matrix_FINBERT.png)

### ROBERTA
![Matrice de Confusion ROBERTA](confusion_matrix_ROBERTA.png)

## Analyse des Erreurs
Une analyse plus approfondie des cas o� les mod�les se trompent peut �tre effectu�e pour identifier les faiblesses.
