# Journal des modifications

## 1.1.0

### Ajouts
- Sous-package `mcfin.blackscholes` : façade `BlackScholesPricer` (formule fermée, Monte Carlo,
  QMC, EDP, arbre), génération de gaussiennes from scratch, EDP de Crank-Nicolson avec lissage de
  Rannacher et payoff moyenné (ordre 2 mesuré), américaines par Brennan-Schwartz et frontière
  d'exercice, simulation de couverture delta (Kamal-Derman, P&L de gamma, Leland), estimation par
  maximum de vraisemblance et estimateurs de range, études de couverture des intervalles de confiance.
- Greeks Black-Scholes jusqu'à l'ordre 3 (charm, speed, zomma, color, ultima, dual delta/gamma, epsilon).
- Tests de propriétés (*hypothesis*) sur les relations d'absence d'arbitrage.
- Exemples 12 à 14 et `docs/BLACK_SCHOLES.md`.
- Outillage : ruff, mypy, couverture de code en CI (seuil 90 %), Makefile, licence MIT, `py.typed`.

### Corrections
- QMC randomisé : l'intervalle de confiance utilise le quantile de Student à R − 1 degrés de liberté
  (`MCResult.dof`) au lieu du quantile gaussien.
- API : `time_index` (date unique → `int`) et `time_indices` (tableau) séparés et typés.

## 1.0.0
- Première version : modèles actions et taux, produits exotiques, Longstaff-Schwartz et borne duale,
  Greeks (bump CRN, LRM, AAD), réduction de variance (dont MLMC), XVA et VaR/ES.
