# Monte Carlo pour la salle des marchés — `mcfin`

[![tests](https://github.com/nicolleremanto/Monte-carlo-for-finance-/actions/workflows/tests.yml/badge.svg)](https://github.com/nicolleremanto/Monte-carlo-for-finance-/actions/workflows/tests.yml)

Librairie Python de **pricing et de gestion des risques par simulation**, couvrant les
méthodes Monte Carlo utilisées sur les desks de dérivés actions, de taux et XVA.
Chaque méthode est **validée contre une formule fermée ou un résultat publié** (suite de
tests `pytest`), et la théorie est détaillée dans [`docs/THEORIE.md`](docs/THEORIE.md).

## Contenu

| Domaine | Implémentation |
|---|---|
| **Aléas** | PCG64, **Sobol brouillé** (Owen, Joe-Kuo) avec QMC randomisé, **pont brownien**, ACP, antithétiques, moment matching, stratification |
| **Modèles actions** | Black-Scholes multi-actifs corrélés, Merton, **Heston (QE-M d'Andersen)** et Euler full truncation, Bates, SABR, **volatilité locale de Dupire** sur nappe **SSVI** sans arbitrage, **LSV calibré par méthode particulaire** (Guyon & Henry-Labordère), **rough Bergomi** (schéma hybride + FFT) |
| **Produits** | Européennes, digitales, paniers, worst-of/best-of, asiatiques, **barrières avec correction de pont brownien**, lookbacks, cliquets, swaps de variance, **autocall Phoenix worst-of**, bermudéennes |
| **Américaines** | **Longstaff-Schwartz** (borne inférieure hors échantillon) + **borne duale d'Andersen-Broadie** par simulations imbriquées |
| **Greeks** | Bump & revalue avec nombres aléatoires communs, **pathwise**, **ratio de vraisemblance**, **AAD** (ruban adjoint vectorisé, écrit de zéro) |
| **Réduction de variance** | Variables de contrôle (β par MCO), **échantillonnage préférentiel** (dérive GHS), **Multilevel Monte Carlo** (Giles) |
| **Taux** | **Hull-White** simulé exactement, Jamshidian ; **LIBOR Market Model** (mesure spot, prédicteur-correcteur, vol abcd, ACP) ; **swaptions bermudéennes** |
| **Risques** | **XVA** : profils EE/PFE, CVA/DVA/FVA, netting, **collatéral avec MPOR** ; **VaR/ES** (FRTB) en revalorisation complète vs delta-gamma, Student-t, bootstrap, contributions d'Euler |
| **Formules fermées** | BS/Black/Bachelier + vol implicite robuste, Heston/Bates (Lewis) + calibration, barrières (Reiner-Rubinstein), BGK, asiatique géométrique, lookback, Merton, Hagan, arbre de Leisen-Reimer |

## Installation

```bash
git clone https://github.com/nicolleremanto/Monte-carlo-for-finance-.git
cd Monte-carlo-for-finance-
pip install -e ".[dev]"      # numpy, scipy (+ pytest, matplotlib)
pytest                       # ~1 min
```

## Exemple rapide

```python
import numpy as np
import mcfin as mc
from mcfin.analytics import heston_price

# Heston, schéma QE d'Andersen, QMC Sobol + pont brownien
model = mc.Heston(spot=100, v0=0.04, kappa=1.5, theta=0.04, xi=0.8, rho=-0.7,
                  rate=0.03, div=0.01, scheme="qe", dt=1/8)
engine = mc.MonteCarloEngine(n_paths=2**16, method="sobol", construction="bridge")
res = engine.price(model, mc.EuropeanOption(strike=100, maturity=2.0))
print(res)                                             # prix, erreur standard, IC 95 %
print(heston_price(100, 100, 2.0, 0.03, 0.01, model.params))  # référence semi-analytique

# Autocall Phoenix worst-of sur 3 indices
corr = np.array([[1, .75, .6], [.75, 1, .65], [.6, .65, 1]])
bs3 = mc.BlackScholes(spot=np.full(3, 100.), vol=np.array([.18, .22, .25]), rate=.03,
                      div=np.array([.03, .025, .02]), corr=corr)
ac = mc.PhoenixAutocall(np.arange(1, 21) / 4, coupon=0.02, coupon_barrier=0.7,
                        protection_barrier=0.6, initial_levels=np.full(3, 100.))
print(mc.MonteCarloEngine(200_000).price(bs3, ac))
print(mc.MonteCarloEngine(100_000).greeks(bs3, ac, {"vol": 0.01}))   # delta, gamma, vega (CRN)
```

## Exemples (`examples/`)

| Script | Ce qu'il montre |
|---|---|
| `01_convergence_reduction_variance.py` | QMC + pont brownien : variance ÷ 3 000 ; variable de contrôle géométrique : ÷ 900 |
| `02_heston_schemas.py` | Biais Euler vs QE-M quand la condition de Feller est violée |
| `03_rough_skew.py` | Skew ATM en loi puissance $T^{H-1/2}$ (rough Bergomi) vs Heston |
| `04_lv_vs_lsv_cliquet.py` | LV et LSV repricent les mêmes vanilles mais pas le même cliquet (smile forward) |
| `05_autocall.py` | Prix, profil de rappel, delta/gamma/vega/corrélation d'un autocall |
| `06_american_bounds.py` | Encadrement LSM / dual Andersen-Broadie |
| `07_greeks.py` | Bump vs LRM vs AAD ; échec pathwise sur digitale ; 7 Greeks Heston en une passe adjointe |
| `08_xva.py` | Profils EE/PFE, CVA/DVA/FVA, effet du netting et du collatéral |
| `09_var_es.py` | VaR 99 % / ES 97,5 % : revalorisation complète vs delta-gamma, Student-t |
| `10_mlmc.py` | MLMC : β ≈ 1 (Euler) vs 2 (Milstein), gain ×164 |
| `11_rates.py` | Hull-White, LMM, swaptions bermudéennes |

```bash
cd examples && python 01_convergence_reduction_variance.py
```

## Quelques résultats

<p align="center">
  <img src="docs/figures/convergence_qmc.png" width="48%"/>
  <img src="docs/figures/heston_bias.png" width="48%"/>
  <img src="docs/figures/rough_skew.png" width="48%"/>
  <img src="docs/figures/xva_exposure.png" width="48%"/>
</p>

| Validation | Monte Carlo | Référence |
|---|---|---|
| Put américain (L-S 2001) | 4,468 ± 0,009 (LSM, 50 dates) | 4,472 |
| Max-call bermudéen (A-B 2004) | [13,90 ; 13,94] (LSM + dual) | [13,892 ; 13,934] |
| Barrières continues (5 types) | pont brownien, 25 dates | Reiner-Rubinstein, < 1 err. std |
| Heston, Feller violé | QE-M, **pas de 1/8 an** | CF de Lewis, < 0,4 err. std |
| Rough Bergomi | pente log-log du skew −0,401 | $H - 1/2 = -0{,}4$ |
| Cliquet ±2 % local | LV 4,39 % / LSV 5,61 % | mêmes vanilles (SSVI) |
| EE d'un swap à une date de reset | XVA Hull-White | swaption de Jamshidian ±0,3 % |
| 7 Greeks Heston | 1 passe AAD (0,25 s) | 14 revalorisations (2,2 s), mêmes valeurs |

## Architecture

```
mcfin/
├── core/            rng.py (PRNG, Sobol, pont brownien, ACP), timegrid.py, results.py
├── market/          curves.py (plate, interpolée, Nelson-Siegel-Svensson), volsurface.py (SSVI, Dupire)
├── analytics/       formules fermées : black_scholes, heston (+calibration), exotics, sabr, lattice
├── models/          equity.py (BS, Merton, Heston/Bates, SABR), local_vol.py (LV, LSV), rough.py
├── products/        vanilla, path_dependent, autocall, american (LSM + Andersen-Broadie)
├── greeks/          aad.py (ruban adjoint), estimators.py (LRM, pricers différentiables)
├── variance_reduction/  importance_sampling.py, mlmc.py
├── rates/           hull_white.py, lmm.py, bermudan.py
├── risk/            xva.py, var.py
└── engine.py        moteur : lots, RQMC, antithétiques, variables de contrôle, IS, Greeks CRN
```

Principes de conception :
* **modèle / produit / moteur découplés** : tout produit se valorise sous tout modèle
  compatible ; le produit déclare ses dates d'observation, le modèle choisit sa grille ;
* modèles en **dataclasses** : un choc de paramètre (`bump_model`) garde la graine, donc des
  **nombres aléatoires communs** pour les Greeks ;
* **erreur standard correcte** pour chaque technique (paires antithétiques, strates,
  répétitions RQMC) ;
* mémoire bornée : simulation par lots, seules les dates d'observation sont stockées.

## Références principales

Glasserman (2003) · Andersen (2008) · Gatheral (2006) · Guyon & Henry-Labordère (2012) ·
Bayer, Friz & Gatheral (2016) · Bennedsen, Lunde & Pakkanen (2017) · Longstaff & Schwartz
(2001) · Andersen & Broadie (2004) · Giles (2008) · Giles & Glasserman (2006) · Brigo &
Mercurio (2006) · Gregory (2020). Liste complète dans [`docs/THEORIE.md`](docs/THEORIE.md).
