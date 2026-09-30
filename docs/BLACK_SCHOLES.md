# Black-Scholes de A à Z

Le modèle de Black-Scholes-Merton (1973) est le socle de toute la finance de marché.
Le sous-package [`mcfin.blackscholes`](../mcfin/blackscholes) le traite comme un objet
d'étude complet : **théorie, quatre méthodes de valorisation, couverture, inférence
statistique et contrôle statistique des estimateurs**. Chaque affirmation de ce document
est vérifiée par un test (`tests/test_blackscholes.py`, `tests/test_properties.py`) ou
mesurée par un script (`examples/12` à `14`).

---

## 1. Le modèle

Sous la probabilité historique $\mathbb P$ :
$$dS_t = \mu S_t\,dt + \sigma S_t\,dW_t \quad\Longrightarrow\quad S_t = S_0\exp\!\Big(\big(\mu - \tfrac{\sigma^2}{2}\big)t + \sigma W_t\Big)$$
(formule d'Itô appliquée à $\ln S_t$). Les log-rendements sont gaussiens i.i.d. ; le taux
sans risque $r$, le rendement du dividende $q$ et la volatilité $\sigma$ sont constants ; le
marché est sans friction et l'on peut négocier en continu.

## 2. Du portefeuille de couverture au prix

**Argument de couverture.** Soit $V(t,S)$ le prix d'une option. Le portefeuille
« option − $\Delta$ actions » avec $\Delta = \partial_S V$ a pour variation, par Itô,
$$d\Pi = \Big(\partial_t V + \tfrac12\sigma^2S^2\partial_{SS}V\Big)dt \quad(\text{le terme en } dW \text{ disparaît}).$$
Sans risque, il doit rapporter le taux sans risque (absence d'opportunité d'arbitrage) :

$$\boxed{\partial_t V + \tfrac12\sigma^2S^2\,\partial_{SS}V + (r-q)S\,\partial_S V - rV = 0,\qquad V(T,S) = \Phi(S).}$$

$\mu$ a disparu : **le prix ne dépend pas de la tendance**.

**Feynman-Kac.** La solution de cette EDP s'écrit comme une espérance :
$V(t,S) = \mathbb E^{\mathbb Q}\big[e^{-r(T-t)}\Phi(S_T)\mid S_t = S\big]$, où, sous $\mathbb Q$,
$dS = (r-q)S\,dt + \sigma S\,dW^{\mathbb Q}$.

**Girsanov.** $\mathbb Q$ s'obtient de $\mathbb P$ par le changement de mesure de densité
$\exp(-\lambda W_T - \frac{\lambda^2}2T)$ avec $\lambda = (\mu - r + q)/\sigma$ (prix de marché du
risque) : sous $\mathbb Q$, les actifs actualisés (dividendes réinvestis) sont des
martingales.

**Formule fermée.** Pour un call, avec $d_{1,2} = \frac{\ln(S/K) + (r-q\pm\sigma^2/2)T}{\sigma\sqrt T}$ :
$$C = S e^{-qT}N(d_1) - K e^{-rT}N(d_2).$$

## 3. Les Greeks et leurs relations

`bs_greeks` fournit 15 sensibilités jusqu'à l'ordre 3 (delta, gamma, vega, theta, rho,
epsilon, vanna, volga, charm, speed, zomma, color, ultima, dual delta, dual gamma),
toutes validées par différences finies (écarts < $10^{-7}$ relatifs). Relations testées
sur 300 jeux de paramètres aléatoires (tests de propriétés *hypothesis*) :

| Relation | Signification |
|---|---|
| $C - P = Se^{-qT} - Ke^{-rT}$ | parité call-put (indépendante du modèle) |
| $(Se^{-qT} - Ke^{-rT})^+ \le C \le Se^{-qT}$ | bornes d'arbitrage |
| $\partial_K C \in [-e^{-rT}, 0]$, $\partial_{KK}C \ge 0$ | monotonie, convexité en strike |
| $\partial_{KK}C = e^{-rT} f_{S_T}(K)$ | Breeden-Litzenberger : la densité risque-neutre se lit dans les prix |
| $\Theta + \frac12\sigma^2S^2\Gamma + (r-q)S\Delta - rV = 0$ | l'EDP elle-même : en delta-neutre, **theta paie le gamma** |

## 4. Quatre méthodes, un seul prix (`BlackScholesPricer.compare`)

Call $S_0 = 100$, $K = 105$, $T = 1$, $r = 3\,\%$, $q = 1\,\%$, $\sigma = 25\,\%$ (exemple 12) :

| Méthode | Principe | Écart à la formule | Remarque |
|---|---|---|---|
| Formule fermée | espérance explicite | — | référence |
| Monte Carlo (200k) | loi des grands nombres | +0,014 (err. std 0,035) | $O(N^{-1/2})$ |
| MC antithétique + contrôle | $\hat Y - \beta(\bar X - \mathbb E X)$, $X = e^{-rT}S_T$ | −0,003 (0,007) | variance ÷ 27 |
| QMC Sobol randomisé | discrépance faible | −0,0005 (0,0005) | ≈ $O(N^{-1})$ |
| EDP Crank-Nicolson | Feynman-Kac | +0,0001 | ordre 2 mesuré : **2,01** |
| Arbre de Leisen-Reimer | Donsker, proba discrète | −5·10⁻⁷ | ordre 2 |

**Détails numériques de l'EDP** (`pde.py`). En log-spot les coefficients sont
constants. Crank-Nicolson brut oscille (le coin du payoff tombe entre deux nœuds :
ordre apparent 0,7) ; Euler implicite est d'ordre 1. Deux corrections standard rétablissent
l'ordre 2 : **moyenner le payoff sur chaque cellule** (analytiquement) et démarrer par
quelques demi-pas implicites (**Rannacher**). Pour l'**américaine**, le problème de
complémentarité $\min(-\partial_\tau V + \mathcal LV,\ V - \Phi) = 0$ est résolu à chaque pas par
l'algorithme de **Brennan-Schwartz** (Gauss tridiagonal projeté) ; on récupère aussi la
**frontière d'exercice** $S^*(\tau)$, qui part de $K$ à l'échéance et décroît avec la
maturité (32,9 à un an pour le put de Longstaff-Schwartz). Le call américain sans
dividende est bien égal à l'européen (Merton 1973).

## 5. Simuler la loi normale (`sampling.py`)

| Méthode | Principe | Acceptation mesurée | Propriété clé |
|---|---|---|---|
| Inversion (Beasley-Springer-Moro) | $X = \Phi^{-1}(U)$ | 1 | monotone en $U$ : seule compatible QMC et stratification ; erreur $3\cdot10^{-9}$ |
| Box-Muller | $R = \sqrt{-2\ln U_1}$, $\Theta = 2\pi U_2$ | 1 | exacte, mais sin/cos |
| Polaire de Marsaglia | rejet dans le disque unité | 0,787 (théorie $\pi/4$) | évite la trigonométrie |
| Rejet depuis Laplace | $f/g \le M = \sqrt{2e/\pi}$ | 0,761 (théorie $1/M$) | illustre l'acceptation-rejet |

Chaque générateur passe un test de Kolmogorov-Smirnov contre $\mathcal N(0,1)$.

## 6. Un prix Monte Carlo est un estimateur (`diagnostics.py`)

On le traite comme en statistique inférentielle : sur 300 répétitions indépendantes,

* **couverture** de l'IC à 95 % : 0,930, IC de Wilson [0,895 ; 0,954] ∋ 0,95 ✓ ;
* **normalité asymptotique** (TCL) des erreurs réduites : Kolmogorov-Smirnov p = 0,94 ✓ ;
* **absence de biais** : test de Student sur les erreurs brutes, p = 0,97 ✓ ;
* **vitesse** : pente log-log de l'erreur standard −0,50 [IC −0,51 ; −0,49] en MC, ≈ −1 en QMC.

Deux pièges statistiques que ces tests ont révélés et que le code corrige :
1. en **QMC randomisé** avec $R$ brouillages, l'erreur est estimée sur $R$ observations :
   l'IC doit utiliser le quantile de **Student à $R-1$ degrés de liberté** (champ `dof` de
   `MCResult`) ;
2. tester le biais sur la moyenne des $Z_r = (\hat V_r - V)/\hat\sigma_r$ est **faux** : pour un
   payoff asymétrique, $\hat V$ et $\hat\sigma$ sont corrélés et $\mathbb E[Z] \neq 0$ même sans
   biais. Il faut tester les erreurs brutes.

Limite honnête : avec 16 brouillages de 256 points, l'estimateur RQMC d'un payoff à coin
n'est pas encore gaussien ; la couverture empirique est ≈ 0,90-0,93. Plus de brouillages
la rétablit (0,933 avec 64) au prix d'une précision moindre.

## 7. La couverture en pratique (`hedging.py`, exemple 13)

On vend l'option à la vol implicite $\sigma_i$ et l'on se couvre en delta à $n$ dates ;
le sous-jacent suit la dynamique historique (tendance $\mu = 12\,\%$).

**Réplication discrète.** Le P&L est d'espérance nulle quelle que soit $\mu$, et son
écart-type décroît en $n^{-1/2}$, conformément à Kamal & Derman (1999) :
$\mathrm{sd} \approx \sqrt{\pi/4}\cdot\text{vega}\cdot\sigma/\sqrt n$.

| rebalancements $n$ | 12 | 52 | 252 | 504 |
|---|---|---|---|---|
| écart-type simulé | 1,90 | 0,93 | 0,43 | 0,30 |
| Kamal-Derman | 2,04 | 0,98 | 0,45 | 0,32 |

**Volatilité mal spécifiée.** Vendue à 25 %, réalisée à 15 %. Couvert à $\sigma_i$, le P&L
vaut en temps continu (El Karoui, Jeanblanc & Shreve 1998)
$$\Pi_T = \tfrac12\int_0^T e^{r(T-t)}\,\Gamma_t S_t^2\,(\sigma_i^2 - \sigma_r^2)\,dt :$$
positif mais **dépendant du chemin** (corrélation simulée avec la formule : 0,98).
Couvert à $\sigma_r$, le gain devient déterministe, $(V(\sigma_i) - V(\sigma_r))e^{rT} = 3{,}99$.
C'est la mécanique d'un desk de volatilité : on achète/vend de la vol implicite et l'on
réalise de la vol par le gamma.

**Coûts de transaction.** Avec une fourchette de 0,4 % et un rebalancement hebdomadaire,
la couverture Black-Scholes perd en moyenne 0,45 ; la volatilité de Leland (1985)
$\sigma_L^2 = \sigma^2\big(1 + \sqrt{2/\pi}\,k/(\sigma\sqrt{\Delta t})\big) = (21{,}1\,\%)^2$ ramène
l'espérance à +0,004.

## 8. Ce que les données permettent d'estimer (`estimation.py`, exemple 14)

L'EMV de $(\mu,\sigma)$ est explicite et l'information de Fisher donne
$\mathrm{Var}(\hat\sigma) \approx \sigma^2/(2n)$ mais $\mathrm{Var}(\hat\mu) \approx \sigma^2/T$ :

| 5 ans, $\sigma = 20\,\%$ | mensuel | hebdo | quotidien | horaire |
|---|---|---|---|---|
| demi-largeur IC 95 % de $\sigma$ | 3,1 pts | 1,6 pt | 0,8 pt | 0,3 pt |
| demi-largeur IC 95 % de $\mu$ | 15 pts | 16 pts | 17 pts | 18 pts |

**La tendance est statistiquement inobservable** ; la volatilité, elle, se mesure d'autant
mieux qu'on échantillonne finement. Heureusement le prix ne dépend que de $\sigma$ : c'est
tout l'intérêt de la valorisation risque-neutre. Couverture empirique des IC vérifiée sur
200 simulations.

Les estimateurs utilisant plus haut et plus bas sont bien plus efficaces que
close-to-close (efficacité mesurée : Parkinson 5,4, Garman-Klass 9,6, Rogers-Satchell 7,9),
mais **biaisés vers le bas** quand l'extremum n'est observé que sur une grille discrète :
biais testé, décroissant avec la finesse de la grille.

## 9. Les limites du modèle — et la suite de la librairie

La volatilité implicite dépend du strike et de la maturité (**smile**), les rendements ont
des queues épaisses, la volatilité est stochastique et « rugueuse ». D'où, dans `mcfin` :
volatilité locale de Dupire (reprice le smile), Heston/Bates (vol stochastique, sauts),
LSV (dynamique du smile), rough Bergomi (skew court terme), SABR (taux) — cf.
[`THEORIE.md`](THEORIE.md).

---

## 10. Lien avec la scolarité à l'ENSAE

Correspondance entre les grands thèmes enseignés (intitulés indicatifs) et le code :

| Thème du cursus | Notions mobilisées | Où dans le code |
|---|---|---|
| Probabilités, calcul stochastique | mouvement brownien, Itô, Girsanov, Feynman-Kac, martingales | `blackscholes/pde.py`, schémas log-Euler martingale, changement de mesure (`importance_sampling.py`) |
| Simulation et méthodes de Monte Carlo | inversion, Box-Muller, acceptation-rejet, variables antithétiques et de contrôle, échantillonnage préférentiel, stratification, QMC, MLMC | `blackscholes/sampling.py`, `core/rng.py`, `engine.py`, `variance_reduction/` |
| Statistique mathématique | EMV, information de Fisher, méthode delta, IC, tests (KS, Student), couverture, bootstrap | `blackscholes/estimation.py`, `blackscholes/diagnostics.py`, `risk/var.py` |
| Économétrie, séries financières | rendements log-normaux, estimateurs de volatilité réalisée | `blackscholes/estimation.py` |
| Analyse numérique, EDP | θ-schémas, stabilité, ordre de convergence, complémentarité linéaire | `blackscholes/pde.py`, `analytics/lattice.py` |
| Optimisation | moindres carrés non linéaires (calibration), régression (Longstaff-Schwartz) | `analytics/heston.py`, `products/american.py` |
| Finance de marché, produits dérivés, gestion des risques | couverture, Greeks, exotiques, XVA, VaR/ES | `products/`, `greeks/`, `risk/` |

## 11. Questions d'entretien (avec réponses courtes)

1. **Pourquoi $\mu$ n'apparaît-il pas dans le prix ?** Le risque est entièrement couvrable
   par le sous-jacent : le portefeuille delta-neutre est sans risque, donc rémunéré à $r$.
   L'exemple 13 le montre : P&L de couverture d'espérance nulle avec $\mu = 12\,\%$.
2. **Que signifie « theta paie le gamma » ?** En delta-neutre,
   $\Theta \approx -\frac12\sigma^2S^2\Gamma$ : le vendeur d'option encaisse le temps et paie la
   convexité. Le P&L quotidien vaut $\frac12\Gamma S^2(\sigma_{réalisée}^2 - \sigma_{implicite}^2)\Delta t$.
3. **Vous vendez de la vol à 25 %, elle se réalise à 15 % : gagnez-vous toujours ?** En
   espérance oui, mais le montant dépend du chemin (poids $\Gamma S^2$) si vous couvrez à la
   vol implicite ; il est verrouillé si vous couvrez à la vol réalisée (que vous ne
   connaissez pas a priori).
4. **Erreur d'un Monte Carlo ? Comment la réduire ?** $\sigma/\sqrt N$ ; antithétiques,
   variables de contrôle (ici le forward, variance ÷ 27), QMC (≈ $N^{-1}$), échantillonnage
   préférentiel pour les queues.
5. **Pourquoi Crank-Nicolson oscille-t-il ?** Il n'est pas L-stable : les hautes fréquences
   du payoff non lisse ne sont pas amorties. Remèdes : Rannacher et lissage du payoff.
6. **Pourquoi l'inversion plutôt que Box-Muller en QMC ?** Seule l'inversion est monotone et
   préserve la structure de faible discrépance, point par point et dimension par dimension.
7. **Peut-on estimer la tendance d'une action sur 10 ans de données quotidiennes ?** Non :
   l'IC de $\mu$ a une demi-largeur ≈ $1{,}96\,\sigma/\sqrt T \approx 12\,\%$, indépendamment de la
   fréquence.
8. **Quand exercer un call américain ?** Jamais sans dividende (sa valeur temps et la
   valeur du report du strike dominent) ; avec dividende, juste avant le détachement si
   assez dans la monnaie. L'EDP le confirme.
9. **Comment vérifier qu'un intervalle de confiance Monte Carlo est correct ?** Par une
   étude de couverture : répéter l'estimation, compter la fréquence d'inclusion du vrai prix
   et la tester (IC de Wilson), plus un test de normalité des erreurs réduites.
10. **Pourquoi Leland augmente-t-il la volatilité ?** Les coûts proportionnels à
    $|\Delta\Delta|$ s'accumulent comme un surcroît de variance d'ordre $k/(\sigma\sqrt{\Delta t})$ :
    on facture ce surcoût dans la prime via une vol plus élevée.

## Références

Black & Scholes (1973), *J. Political Economy* · Merton (1973), *Bell J. Economics* ·
Harrison & Pliska (1981) · Leland (1985), *J. Finance* · El Karoui, Jeanblanc & Shreve
(1998), *Math. Finance* · Kamal & Derman (1999), Goldman Sachs QS Research Notes ·
Brennan & Schwartz (1977), *J. Finance* · Rannacher (1984), *Numer. Math.* · Leisen &
Reimer (1996), *Appl. Math. Finance* · Box & Muller (1958) · Marsaglia & Bray (1964) ·
Moro (1995), *Risk* · Parkinson (1980), Garman & Klass (1980), Rogers & Satchell (1991) ·
L'Ecuyer, Munger & Tuffin (2010), *Monte Carlo Methods Appl.* · Glasserman (2003) ·
Lamberton & Lapeyre, *Introduction au calcul stochastique appliqué à la finance* ·
Shreve, *Stochastic Calculus for Finance II*.
