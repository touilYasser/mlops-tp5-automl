# Modele manuel ou modele AutoML ? (TP05)

Mesures realisees localement le 31/10/2026 avec Python 3.13 et MLflow 3.17.2.

| Critere | Manuel (rf 500/16) | AutoML (FLAML 90 s) |
|---|---:|---:|
| Famille de modele | foret aleatoire | xgboost |
| F1 sur le jeu de test | 0.4276 | 0.4400 |
| Temps de recherche machine | 0.9 s | 90.0 s |
| Temps humain (estimation) | environ 2 h | environ 20 min |
| Latence mediane (ms) | 35.1 | 6.6 |
| Latence p95 (ms) | 53.2 | 7.7 |
| Taille du fichier (Mo) | 31.48 | 0.04 |
| Explicabilite | importance des variables, 500 arbres, profondeur 16 | importance des variables et hyperparametres XGBoost |

## Decision

Le modele AutoML est retenu pour la production : il obtient le meilleur F1, une latence nettement inferieure
et un artefact beaucoup plus petit. Le modele manuel reste conserve dans le registre comme reference et
solution de repli. L'alias `production` permet de changer de modele sans modifier le client.
