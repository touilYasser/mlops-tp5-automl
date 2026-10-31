# Rapport TP05 - AutoML et deploiement

**Etudiant :** Yasser Touil  
**Filiere :** 5IASD  
**Groupe :** G6  
**Date des mesures :** 31/10/2026

## Resultats techniques

Le modele AutoML FLAML a retenu XGBoost avec un budget de 90 secondes. Il a obtenu un F1 de
0.4400 sur le jeu de test et a ete enregistre sous `fraude-automl`, version 1. Le modele manuel
RandomForest (500 arbres, profondeur 16) a obtenu 0.4276 et est enregistre sous `fraude-manuel`,
version 2. Les deux versions sont exposees par l'alias `production`.

Les services ont repondu sur les ports 8100 (AutoML) et 8101 (manuel). Les mesures du client sont :
AutoML, mediane 6.6 ms et p95 7.7 ms ; manuel, mediane 35.1 ms et p95 53.2 ms. Une requete ne
contenant que `montant` est correctement rejetee avec les colonnes manquantes `anciennete`,
`freq_24h` et `nuit`.

## Comparaison

Voir [tp5_comparaison.md](./tp5_comparaison.md).

## Questions

### 1. Choix de production

Je retiens AutoML : son F1 est meilleur, sa latence mediane et son p95 sont plus faibles, et son
artefact est beaucoup plus petit. La taille facilite le deploiement et l'explicabilite reste possible
avec l'importance des variables et les hyperparametres journalises dans MLflow.

### 2. Cout d'un endpoint cloud

Un endpoint cloud reste facture pour la machine deployee meme sans requete. Le calcul doit utiliser
le tarif horaire de la machine choisie, multiplie par environ 730 heures par mois, en indiquant la
date de consultation du tarif. Les couts de stockage, de prediction et de reseau doivent etre ajoutes
separement lorsqu'ils s'appliquent.

### 3. Latence locale et cloud

En local, l'appel reste dans la machine et ne traverse pas Internet. Dans le cloud, il faut ajouter
la resolution reseau, le transport, le proxy, la file d'attente et le temps de calcul de l'instance.
Les mesures locales de ce TP donnent 6.6 ms de mediane pour AutoML et 35.1 ms pour le modele manuel.

### 4. Changement de version du moteur AutoML

Une nouvelle version peut modifier les algorithmes, les hyperparametres, la serialization ou les
resultats. Le risque est une regression de F1, de latence ou de compatibilite. Il faut donc figer les
versions, conserver l'artefact exporte et valider chaque nouvelle version avant promotion.

### 5. Reversibilite

Je conserve le modele exporte, les donnees de reference, le code d'entrainement, les parametres,
les metriques et l'environnement Python dans Git et MLflow. Une procedure de restauration doit
permettre de redemarrer le service localement et de repromouvoir l'ancienne version par son alias.

### 6. skops ou cloudpickle

`skops` reduit le risque d'execution arbitraire lors du chargement et permet une liste explicite des
types de confiance. `cloudpickle` est plus flexible pour serialiser le modele produit par FLAML,
notamment lorsque la famille ou les objets internes varient. Il reste dangereux avec une source non
fiable ; il est acceptable ici car l'artefact est produit et servi dans notre environnement controle.
