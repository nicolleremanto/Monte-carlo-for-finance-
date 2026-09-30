# Théorie et choix d'implémentation

Ce document explique **pourquoi** chaque méthode de `mcfin` existe, **comment** elle est
implémentée et **ce qu'il faut savoir en dire en entretien**. Les renvois `module.py`
pointent vers le code correspondant.

---

## 1. Fondements du Monte Carlo

On veut $V_0 = \mathbb{E}^{\mathbb{Q}}[D(0,T)\,\Phi(X)]$ où $\Phi$ est le payoff, $X$ la
trajectoire des facteurs de risque et $D$ le facteur d'actualisation (ou l'inverse du
numéraire). L'estimateur $\hat V_N = \frac1N\sum_{i=1}^N Y_i$ vérifie :

* **loi des grands nombres** : $\hat V_N \to V_0$ p.s. ;
* **TCL** : $\sqrt N(\hat V_N - V_0) \Rightarrow \mathcal N(0, \sigma^2)$, d'où l'intervalle
  de confiance $\hat V_N \pm 1{,}96\,\hat\sigma/\sqrt N$ (`core/results.py`, `MCResult.ci`).

L'erreur décroît en $N^{-1/2}$ **quelle que soit la dimension** — c'est l'avantage décisif
sur les EDP dès 3-4 facteurs (paniers, taux multi-facteurs, XVA). Diviser l'erreur par 10
coûte 100 fois plus de trajectoires : d'où l'importance de la réduction de variance.

Avec un schéma de discrétisation de pas $h$, l'erreur quadratique moyenne se décompose :

$$\text{MSE} = \underbrace{(\mathbb E[\hat Y_h] - V_0)^2}_{\text{biais}^2 = O(h^{2\alpha})} + \underbrace{\sigma^2/N}_{\text{variance}}$$

Pour une précision $\varepsilon$, un MC standard avec Euler ($\alpha = 1$) coûte
$O(\varepsilon^{-3})$ ; le MLMC ramène ce coût à $O(\varepsilon^{-2}(\log\varepsilon)^2)$ (§7).

**Estimation de l'erreur** (`engine.py`) — un point souvent mal traité :

| Technique | Erreur standard correcte |
|---|---|
| pseudo-aléatoire | $\hat\sigma/\sqrt N$ |
| antithétiques | écart-type des **moyennes de paires** $/\sqrt{N/2}$ (les deux tirages d'une paire sont corrélés) |
| stratification | $\sqrt{\sum_j p_j^2 s_j^2/n_j}$ |
| QMC randomisé | écart-type des $R$ estimateurs indépendants (brouillages) $/\sqrt R$, IC de Student à $R-1$ ddl |
| moment matching | tirages d'un groupe rendus dépendants : écart-type sur $R$ groupes renormalisés indépendamment (Student à $R-1$ ddl) |

---

## 2. Génération des aléas (`core/rng.py`)

### Pseudo-aléatoire
PCG64 (numpy) : période $2^{128}$, flux indépendants, reproductible par graine. La graine
fixe est **indispensable** aux Greeks par différences finies (nombres aléatoires communs).

### Quasi-Monte Carlo : suites de Sobol
Les suites à discrépance faible remplissent $[0,1]^d$ plus uniformément que le hasard.
L'inégalité de Koksma-Hlawka borne l'erreur par $V(f)\,D^*_N$ avec
$D^*_N = O((\log N)^d/N)$ : presque $O(1/N)$ en pratique quand la *dimension effective*
est faible. Nous utilisons Sobol avec les nombres directeurs de Joe & Kuo (2008) et le
**brouillage d'Owen** (1997), qui rend chaque point uniforme (estimateur sans biais) et
permet d'estimer l'erreur par répétition sur $R$ brouillages indépendants (RQMC).

### Construction du brownien : pont brownien et ACP
Les premières coordonnées de Sobol sont les mieux réparties. On les affecte donc aux
directions qui portent le plus de variance :

* **pont brownien** : on tire d'abord $W(T)$, puis les points milieux conditionnellement
  aux extrémités :
  $W(u)\mid W(s),W(t) \sim \mathcal N\!\left(\frac{(t-u)W(s)+(u-s)W(t)}{t-s},\ \frac{(u-s)(t-u)}{t-s}\right)$ ;
* **ACP** : vecteurs propres de la covariance $\min(t_i,t_j)$ par valeur propre décroissante
  (optimal en variance expliquée, coût $O(n^2)$).

En pseudo-aléatoire ces constructions ne changent rien (même loi) ; en QMC elles changent
tout : sur l'asiatique de l'exemple 01, **Sobol + pont brownien divise la variance par
≈ 3 000** par rapport au pseudo-aléatoire à nombre de trajectoires égal.

### Loi de Poisson par inversion
Les sauts (Merton, Bates) sont tirés par inversion de la fonction de répartition
(`poisson_inverse`), ce qui garde **une uniforme par variable** (compatible QMC) et est
~100× plus rapide que `scipy.stats.poisson.ppf`.

---

## 3. Discrétisation des EDS

Pour $dX = a(X)dt + b(X)dW$ :

* **Euler** : $X_{n+1} = X_n + a\,h + b\,\Delta W$ — ordre fort ½, ordre faible 1 ;
* **Milstein** : $+\ \tfrac12 b b'(\Delta W^2 - h)$ — ordre fort 1 ;
* **log-Euler** pour un actif : $\ln S_{n+1} = \ln S_n + (\mu - \tfrac12\sigma_n^2)h + \sigma_n\Delta W$.
  Avec $\sigma_n$ évalué en début de pas, $\mathbb E[S_{n+1}\mid\mathcal F_n] = S_n e^{\mu h}$
  **exactement** : le schéma est martingale, donc les forwards sont reproduits sans biais.

Chaque fois que c'est possible, on **simule exactement** : GBM, Merton (aux dates de la
grille), Hull-White (vecteur gaussien $(x_t, \int x)$). Le moteur ne stocke que les dates
d'observation du produit, et découpe les trajectoires en lots pour borner la mémoire.

---

## 4. Modèles actions (`models/`)

### 4.1 Black-Scholes multi-actifs
$dS_i/S_i = (r-q_i)dt + \sigma_i dW_i$, $d\langle W_i,W_j\rangle = \rho_{ij}dt$ ; corrélation par
Cholesky ($Z_{corr} = L Z$). Simulation exacte du log-prix. Courbes de taux et de dividende
générales : la dérive par pas vaut $\ln\frac{F(t_{i+1})}{F(t_i)}$ avec $F(t) = S_0 D_q(t)/D_r(t)$.

### 4.2 Merton et Bates (sauts)
$\ln J \sim \mathcal N(\mu_J,\delta^2)$, sauts de Poisson($\lambda$), compensateur
$-\lambda\bar k\,dt$ avec $\bar k = e^{\mu_J+\delta^2/2}-1$. Somme de $N$ sauts gaussiens
$= N\mu_J + \sqrt N\,\delta Z$. Référence : série de Merton pondérée par Poisson($\lambda(1+\bar k)T$).

### 4.3 Heston et le schéma QE d'Andersen
$$dS/S = (r-q)dt + \sqrt{v}\,dW_S,\qquad dv = \kappa(\theta - v)dt + \xi\sqrt v\,dW_v,\qquad d\langle W_S,W_v\rangle = \rho\,dt$$

**Pricing de référence** (`analytics/heston.py`) : fonction caractéristique dans la
formulation d'Albrecher et al. (2007) (« little Heston trap », sans discontinuité de branche)
et formule de Lewis (2000) :
$$C = S_0e^{-qT} - \frac{\sqrt{FK}e^{-rT}}{\pi}\int_0^\infty \frac{\mathrm{Re}\big[e^{iuk}\varphi(u-i/2)\big]}{u^2+1/4}\,du,\quad k=\ln(F/K)$$
intégrée par Gauss-Legendre par panneaux (précision $10^{-12}$ contre une quadrature
adaptative). Calibration Levenberg-Marquardt sur les erreurs de vol pondérées par la vega.

**Pourquoi Euler échoue** : la CIR peut devenir négative ; les corrections (troncature,
réflexion) biaisent, surtout quand la **condition de Feller** $2\kappa\theta > \xi^2$ est
violée (cas typique des calibrations actions). *Full truncation* (Lord et al. 2010) est la
meilleure variante d'Euler mais reste biaisée (exemple 02 : +4,5 avec 1 pas/an, +0,10 avec
64 pas/an).

**QE (Quadratic-Exponential, Andersen 2008)** : on apparie les deux premiers moments de
la loi conditionnelle (χ² non centrée) de $v_{t+\Delta}$, $m$ et $s^2$, avec $\psi = s^2/m^2$ :
* si $\psi \le 1{,}5$ : $v' = a(b+Z)^2$ avec $b^2 = 2/\psi - 1 + \sqrt{2/\psi}\sqrt{2/\psi-1}$, $a = m/(1+b^2)$ ;
* sinon : masse $p = (\psi-1)/(\psi+1)$ en 0 et queue exponentielle de paramètre $\beta = (1-p)/m$.

Le log-spot utilise l'intégrale de $v$ par trapèzes et l'identité
$\int\sqrt v\,dW_v = (v_{t+\Delta} - v_t - \kappa\theta\Delta + \kappa\int v)/\xi$ ;
la **correction de martingale** remplace $K_0$ pour que $\mathbb E[S_{t+\Delta}/S_t] = e^{(r-q)\Delta}$
exactement. Résultat : biais indétectable même avec **un pas par an**.

### 4.4 SABR
$dF = \alpha F^\beta dW_1$, $d\alpha = \nu\alpha dW_2$. $\alpha$ est lognormal (simulé exactement),
$F$ par Euler **absorbé en 0** ($\beta<1$). Comparaison à la formule asymptotique de Hagan
(2002) : écarts ≤ 12 pb de vol sur l'exemple testé (l'approximation se dégrade pour les
strikes bas et les maturités longues).

### 4.5 Volatilité locale de Dupire
Le seul modèle markovien à un facteur qui reprice **toutes** les vanilles :
$\sigma_{loc}^2(K,T) = \frac{\partial_T C}{\frac12K^2\partial_{KK}C}$ (à taux nuls). En variance
totale implicite $w(y,T) = \sigma_{imp}^2 T$, $y = \ln(K/F_T)$ (Gatheral 2006) :
$$\sigma_{loc}^2 = \frac{\partial_T w}{1 - \frac yw\partial_y w + \frac14\left(-\frac14 - \frac1w + \frac{y^2}{w^2}\right)(\partial_y w)^2 + \frac12\partial_{yy}w}$$
Le dénominateur est positif ssi la nappe est sans arbitrage papillon ; le numérateur ssi
elle est sans arbitrage calendaire. On part donc d'une **nappe SSVI** (Gatheral & Jacquier
2014) :
$w(y,\theta_T) = \frac{\theta_T}2\left[1 + \rho\varphi y + \sqrt{(\varphi y+\rho)^2 + 1-\rho^2}\right]$, avec
$\varphi(\theta) = \eta\,\theta^{-\gamma}(1+\theta)^{\gamma-1}$, sans arbitrage statique si
$\theta_T$ croît, $0<\gamma\le\frac12$ et $\eta(1+|\rho|)\le 2$. Toutes les dérivées sont
analytiques (`market/volsurface.py`).

**Limite** : la LV prédit un smile *forward* trop plat. Elle sous-évalue les produits qui
dépendent de la volatilité future : cliquets, forward-starts, options sur variance.

### 4.6 LSV (volatilité locale-stochastique) et méthode particulaire
$dS/S = (r-q)dt + L(t,S)\sqrt{v}\,dW_S$, avec $v$ de type Heston. Par le théorème de
projection de Gyöngy (1986), le modèle reprice les vanilles ssi
$$L^2(t,K)\;\mathbb E[v_t \mid S_t = K] = \sigma_{loc}^2(t,K).$$
L'espérance conditionnelle dépend de la loi jointe de $(S_t, v_t)$, donc de $L$ elle-même :
c'est une EDS de **McKean-Vlasov**. Guyon & Henry-Labordère (2012) la résolvent en une seule
simulation de $N$ particules, en estimant $\mathbb E[v\mid S=K]$ par régression à noyau
(Nadaraya-Watson), fenêtre $h = 1{,}5\,\sigma_{loc}(t,S_0)\sqrt{\max(t,1/4)}\,N^{-1/5}$
(`models/local_vol.py`).

Résultats (exemple 04) : LV et LSV repricent la nappe à quelques pb près, mais le cliquet
(cap/floor locaux ±2 %) vaut **4,39 % en LV contre 5,61 % en LSV** : c'est la valeur du
smile forward, et l'argument pour le LSV sur un desk d'exotiques.

*Limite observée* : avec une vol-of-vol forte et Feller très violé
($2\kappa\theta/\xi^2 \approx 0{,}4$), l'aile droite courte garde un biais de discrétisation
(≈ 60 pb avec $\Delta t = 1/100$, divisé par deux à $1/400$). En pratique on applique un
*mixing* ($\xi$ réduit) et des pas plus fins.

### 4.7 Rough Bergomi
Gatheral, Jaisson & Rosenbaum (2018) : la log-volatilité réalisée se comporte comme un
mouvement brownien fractionnaire d'indice $H\approx 0{,}1$. Bayer, Friz & Gatheral (2016) :
$$v_t = \xi_0(t)\exp\!\left(\eta Y_t - \tfrac{\eta^2}2 t^{2H}\right),\qquad Y_t = \sqrt{2H}\int_0^t (t-s)^{H-1/2}dW_s.$$
$Y$ n'est pas markovien : pas d'EDP, le Monte Carlo est la méthode naturelle. **Schéma
hybride** (Bennedsen, Lunde & Pakkanen 2017), $\kappa = 1$, $\alpha = H - \frac12$ : le noyau
est intégré exactement sur la dernière cellule (vecteur gaussien 2D $(\Delta W_i, \tilde W_i)$) et
approché par une fonction en escalier ailleurs, aux points optimaux
$b_k = \left(\frac{k^{\alpha+1}-(k-1)^{\alpha+1}}{\alpha+1}\right)^{1/\alpha}$. La somme de Riemann
est une convolution calculée par FFT, en $O(n\log n)$.

Résultat (exemple 03) : le skew ATM suit une **loi puissance de pente −0,401**
(théorie : $H-\frac12 = -0{,}4$), alors que le skew de Heston est plat aux maturités courtes.

---

## 5. Produits (`products/`)

### Asiatiques
Moyenne arithmétique : pas de formule fermée. La **moyenne géométrique** est lognormale,
$\ln G \sim \mathcal N\!\big(\ln S_0 + (r-q-\frac{\sigma^2}2)\bar t,\ \frac{\sigma^2}{n^2}\sum_{i,j}\min(t_i,t_j)\big)$,
d'où une formule fermée utilisée comme **variable de contrôle** (Kemna & Vorst 1990).
Corrélation > 0,999 avec l'arithmétique : variance divisée par ≈ 900.

### Barrières et correction de pont brownien
Surveiller une barrière continue aux seules dates de la grille surestime la survie (biais en
$O(\sqrt{\Delta t})$). Sachant les extrémités $x_i, x_{i+1}$ (log-distances à la barrière), le
brownien entre deux dates est un pont, et
$$\mathbb P(\text{franchissement}) = \exp\!\left(-\frac{2\,x_i\,x_{i+1}}{\int_{t_i}^{t_{i+1}}\sigma^2dt}\right).$$
On multiplie les probabilités de survie (estimateur conditionnel, **sans biais** en GBM et
de variance réduite). Réciproquement, pour valoriser une barrière *discrète* avec une formule
continue, on décale la barrière : $H e^{\pm\beta\sigma\sqrt{\Delta t}}$ avec
$\beta = -\zeta(\frac12)/\sqrt{2\pi}\approx 0{,}5826$ (Broadie, Glasserman & Kou 1997). Les tests vérifient
les deux.

### Autocall Phoenix
Coupons conditionnels (digitales sur le worst-of, avec mémoire), rappel anticipé si le
worst-of dépasse 100 %, put down-and-in à maturité. L'investisseur est **vendeur de
volatilité et acheteur de corrélation** ; la banque porte les positions inverses, plus un
gamma digital près des barrières. Les niveaux initiaux doivent être **figés**
(`initial_levels`) : sinon un choc de spot est neutralisé par la normalisation $S_t/S_0$,
et le delta sort nul (erreur classique).

### Bermudéennes : Longstaff-Schwartz et dualité
**LSM** (2001) : rétro-induction, régression MCO des flux futurs actualisés sur une base
de fonctions de l'état (monômes de degré ≤ 3 + payoff), restreinte aux trajectoires dans la
monnaie. La règle estimée est appliquée à des trajectoires **indépendantes** : le prix est
alors une borne inférieure non biaisée (pas de biais de prévoyance).

**Borne duale** (Rogers 2002, Haugh & Kogan 2004, Andersen & Broadie 2004) :
$V_0 \le \mathbb E[\max_k(\tilde h_k - M_k)]$ pour toute martingale $M$ nulle en 0. On construit
$M$ à partir de la règle LSM ($M_k = M_{k-1} + \tilde L_k - \mathbb E_{k-1}[\tilde L_k]$), les
espérances conditionnelles étant estimées par **simulations imbriquées**. L'écart de dualité
mesure la sous-optimalité de la règle. Max-call 2 actifs : intervalle obtenu
[13,90 ; 13,94], contre [13,892 ; 13,934] dans l'article.

---

## 6. Greeks (`engine.greeks`, `greeks/`)

| Méthode | Principe | Avantages | Limites |
|---|---|---|---|
| Bump & revalue + CRN | $\frac{V(\theta+h)-V(\theta-h)}{2h}$ avec la même graine | générique, simple | 2 pricings par paramètre ; biais $O(h^2)$ |
| Pathwise | $\mathbb E[\partial_\theta Y]$ | variance minimale, sans biais | payoff lipschitzien requis |
| LRM | $\mathbb E[Y\,\partial_\theta\ln p_\theta]$ | payoffs discontinus | variance forte |
| AAD | pathwise calculé en mode adjoint | **tous** les Greeks pour ≈ 4× un pricing | mémoire (ruban), payoff lisse |

**Pourquoi les CRN** : sans graine commune, $\mathrm{Var}[\hat V(\theta+h)-\hat V(\theta-h)] = O(1/N)$
et l'estimateur a une variance $O(1/(Nh^2))$ qui explose quand $h\to0$ ; avec CRN, la
différence est $O(h)$ trajectoire par trajectoire et la variance reste $O(1/N)$.

**Scores LRM pour un GBM** : delta $z_1/(S_0\sigma\sqrt{\Delta t_1})$ ; vega
$\sum_i[(z_i^2-1)/\sigma - z_i\sqrt{\Delta t_i}]$.

**AAD** (`greeks/aad.py`) : ruban d'opérations vectorisées numpy ; chaque nœud stocke ses
fonctions vecteur-jacobien ; la passe arrière propage $\bar x = \partial V/\partial x$. Principe du
gradient bon marché : coût borné indépendamment du nombre de paramètres. Démonstration
(exemple 07) : 7 sensibilités Heston en une passe adjointe, identiques aux différences
finies à 4 chiffres. Deux subtilités :
1. digitale : la dérivée pathwise est nulle p.s., l'AAD « brut » renvoie 0 → lissage
   sigmoïde (biais $O(\varepsilon^2)$) ou LRM ;
2. Heston avec Feller violé : $\partial\sqrt v/\partial v = 1/(2\sqrt v)$ explose près de 0 :
   l'estimateur pathwise reste valide mais sa variance devient très élevée.

---

## 7. Réduction de variance

* **Antithétiques** : $(Z,-Z)$ ; efficace si le payoff est monotone en $Z$ (gain ≈ 2 sur un call,
  nul sur un straddle).
* **Variables de contrôle** : $\hat Y_{cv} = \bar Y - \hat\beta^\top(\bar X - \mu_X)$ avec
  $\hat\beta$ par MCO ; facteur de réduction $1/(1-R^2)$. Contrôles : asiatique géométrique,
  forward du sous-jacent, vanille Heston analytique...
* **Stratification** de $W(T)$ (via le pont brownien) : supprime la variance inter-strates.
* **Échantillonnage préférentiel** : changement de dérive gaussien, rapport de vraisemblance
  $e^{-\mu\cdot Z + |\mu|^2/2}$ ; dérive de Glasserman-Heidelberger-Shahabuddin (1999)
  $\mu^* = \arg\max_z[\ln G(z) - |z|^2/2]$. Call très en dehors de la monnaie : variance
  divisée par **9 000**.
* **QMC + pont brownien** : cf. §2.
* **Multilevel Monte Carlo** (Giles 2008) :
  $\mathbb E[P_L] = \mathbb E[P_0] + \sum_{l=1}^L\mathbb E[P_l - P_{l-1}]$, les corrections étant
  simulées avec le **même** brownien (pas fin et pas grossier couplés). Si
  $\mathrm{Var}[P_l-P_{l-1}] = O(h_l^\beta)$, l'allocation optimale $N_l\propto\sqrt{V_l/C_l}$ donne
  un coût $O(\varepsilon^{-2})$ pour $\beta>1$ (Milstein) et $O(\varepsilon^{-2}\log^2\varepsilon)$
  pour $\beta=1$ (Euler). Mesuré : β ≈ 1,0 (Euler) et 1,9 (Milstein), gain ×164 à ε = 0,005.

---

## 8. Taux (`rates/`)

### Hull-White 1 facteur
$dr = (\theta(t) - ar)dt + \sigma dW$ ; $r = x + \varphi$ avec $x$ Ornstein-Uhlenbeck et
$\varphi(t) = f(0,t) + \frac{\sigma^2}{2a^2}(1-e^{-at})^2$ : la courbe initiale est reproduite.
Zéro-coupons affines :
$$P(t,T) = \frac{P(0,T)}{P(0,t)}\exp\!\Big(-B x_t - \tfrac{\sigma^2}{4a}(1-e^{-2at})B^2 - B\tfrac{\sigma^2}{2a^2}(1-e^{-at})^2\Big),\quad B = \tfrac{1-e^{-a(T-t)}}a.$$
Simulation **exacte** du couple $(x_t, \int_0^t x_s ds)$, d'où le déflateur
$D(t) = P(0,t)\,e^{-\int x - \frac{\sigma^2}{2a^2}[\dots]}$. Swaptions européennes par
**Jamshidian** : l'obligation à coupons est monotone en $x$ ; on résout $\sum c_iP(T_0,T_i;x^*) = 1$
et la swaption devient une somme de puts sur zéro-coupons.

### LIBOR / Euribor Market Model
Forwards $L_i$ lognormaux sous leur mesure forward (formule de Black pour les caplets).
Sous la mesure spot (numéraire : compte courant discret) :
$$\frac{dL_i}{L_i} = \sigma_i\sum_{j=\eta(t)}^{i}\frac{\tau_j\rho_{ij}\sigma_jL_j}{1+\tau_jL_j}dt + \sigma_i dW_i.$$
Log-Euler **prédicteur-correcteur** (Hunter, Jäckel & Joshi 2001), covariance intégrée exacte
par pas, réduction à $k$ facteurs par ACP avec renormalisation des variances. Vol « abcd »
de Rebonato, corrélation exponentielle. Contrôles : $\mathbb E[1/B(T_k)] = P(0,T_k)$ (numéraire),
caplets = Black, swaptions ≈ formule de Rebonato (≈ 1 % en plein rang). Réduire le nombre
de facteurs augmente les corrélations effectives, donc les swaptions : c'est visible dans
les tests. Après la réforme des IBOR, l'Euribor subsiste ; pour les taux RFR composés
(€STR, SOFR), l'extension est le *Forward Market Model* (Lyashenko & Mercurio 2019).

### Swaptions bermudéennes
LSM sur les swaps co-terminaux, flux déflatés par le numéraire du modèle, régressions en
valeur « $t_k$ ». Contrôles : une seule date d'exercice redonne Jamshidian ; le prix
dépasse le maximum des européennes co-terminales (prime de switch).

---

## 9. XVA (`risk/xva.py`)

1. **Diffusion** des facteurs (Hull-White exact) sur la grille d'exposition + dates de
   fixing + dates décalées de la MPOR.
2. **Revalorisation** de chaque swap à chaque date, analytique (pas de Monte Carlo imbriqué),
   y compris le coupon variable déjà fixé.
3. **Netting** : somme des valeurs par contrepartie avant la partie positive.
4. **Collatéral** (CSA bilatéral) : $C(t) = (V(t-\delta)-H_c)^+ - (-V(t-\delta)-H_b)^+$ ;
   exposition $V(t) - C(t)$ ; $\delta$ = marge de risque (MPOR, 10 jours ouvrés).
5. **Profils** : EE, ENE, PFE à 95 %, EPE et EEPE (Bâle).
6. **Ajustements** (unilatéraux discrétisés, version « first-to-default » optionnelle) :
$$\text{CVA} = (1-R_c)\sum_i \mathbb E[D(t_i)E^+(t_i)]\,\big(S_c(t_{i-1})-S_c(t_i)\big)\,S_b(t_{i-1})$$
   DVA symétrique sur $E^-$ ; FVA = FCA + FBA avec un spread de financement. Survie
   $S(t) = e^{-\lambda t}$ avec $\lambda \approx s_{CDS}/(1-R)$ (triangle du crédit).

Validation : pour un swap payeur, $\mathbb E[D(T_k)V^+(T_k)]$ à une date de reset **est** la
swaption européenne sur le swap restant ; les tests le vérifient contre Jamshidian (±2 %).
Exemple 08 : netting −19 % de CVA, CSA −68 %. Hypothèse simplificatrice : indépendance
crédit/marché (pas de *wrong-way risk*).

---

## 10. VaR et Expected Shortfall (`risk/var.py`)

$\text{VaR}_\alpha = -q_{1-\alpha}(\text{P\&L})$ ;
$\text{ES}_\alpha = -\mathbb E[\text{P\&L}\mid\text{P\&L}\le -\text{VaR}_\alpha]$. La FRTB remplace la VaR 99 %
par l'**ES 97,5 %** (cohérente, sous-additive, sensible à la forme de la queue) avec des
horizons de liquidité (10 jours et plus).

* Facteurs de risque : log-rendements gaussiens ou **Student-t multivariés** (même
  covariance, queues épaisses) + chocs de vol implicite corrélés négativement au spot.
* **Revalorisation complète** vs **delta** vs **delta-gamma-vega** : sur un book court en
  gamma, l'approximation delta surestime la VaR de ≈ 11 % ; delta-gamma est à ≈ 5 %.
* **IC bootstrap** de la VaR et de l'ES ; **contributions d'Euler**
  $\text{ES}_i = -\mathbb E[\text{P\&L}_i\mid\text{queue}]$ : allocation additive exacte (homogénéité
  de degré 1 de l'ES).

---

## 11. Tableau de validation

| Test | Méthode MC | Référence | Résultat |
|---|---|---|---|
| Call européen BS | Sobol + pont brownien, $2^{16}$ trajectoires | Black-Scholes | erreur ≈ 0,0004, err. std 0,0008 |
| Heston ($\xi = 0{,}8$, Feller violé) | QE-M, pas = 1/8 an | Lewis (CF) | < 0,4 err. std |
| Bates, Merton | exact / QE | séries / CF | < 1,5 err. std |
| Barrières (5 types) | pont brownien, 25 dates | Reiner-Rubinstein | < 1 err. std |
| Lookback flottant | 252 dates + BGK | Goldman-Sosin-Gatto | < 1,2 err. std |
| Put américain | LSM, 50 dates | Longstaff-Schwartz (4,472) | 4,468 ± 0,009 |
| Max-call bermudéen | LSM + dual A-B | Andersen-Broadie [13,892 ; 13,934] | [13,90 ; 13,94] |
| Vol locale SSVI | log-Euler, Δt = 1/100 | nappe SSVI | < 15 pb (T ≥ 1) |
| LSV particulaire | 50 000 particules | nappe SSVI | < 20 pb ($\xi = 0{,}3$) |
| Rough Bergomi | schéma hybride | $\mathrm{Var}\,Y_t = t^{2H}$, pente du skew | pente −0,401 (−0,4) |
| HW : caplet, swaption | exact | formules fermées, Jamshidian | < 1 err. std |
| LMM : numéraire, caplets | prédicteur-correcteur | $P(0,T)$, Black | < 1,5 err. std |
| AAD Heston | ruban adjoint | différences finies $h\to0$ | identique (Feller respecté) |
| EE d'un swap | XVA HW | swaption Jamshidian | ±0,3 % |

---

## 12. Limites et extensions naturelles

* Dividendes discrets (modèle *spot/forward* de Bühler), repo, multi-devises (quanto).
* Multi-courbes (OIS/€STR pour l'actualisation, Euribor pour la projection).
* Modèles hybrides actions-taux (LSV + Hull-White), vol stochastique sur les taux (SABR-LMM).
* *Wrong-way risk*, KVA/MVA (marge initiale via SIMM), XVA Greeks par AAD.
* Parallélisation GPU (CuPy/JAX) : le moteur est déjà vectorisé par trajectoires.

---

## Références

* P. Glasserman, *Monte Carlo Methods in Financial Engineering*, Springer, 2003.
* P. Jäckel, *Monte Carlo Methods in Finance*, Wiley, 2002.
* L. Andersen, « Simple and efficient simulation of the Heston stochastic volatility model », *J. Comp. Finance*, 2008.
* R. Lord, R. Koekkoek, D. van Dijk, « A comparison of biased simulation schemes for stochastic volatility models », *Quant. Finance*, 2010.
* H. Albrecher, P. Mayer, W. Schoutens, J. Tistaert, « The little Heston trap », *Wilmott*, 2007.
* A. Lewis, *Option Valuation under Stochastic Volatility*, 2000.
* J. Gatheral, *The Volatility Surface*, Wiley, 2006 ; J. Gatheral, A. Jacquier, « Arbitrage-free SVI volatility surfaces », *Quant. Finance*, 2014.
* B. Dupire, « Pricing with a smile », *Risk*, 1994 ; I. Gyöngy, « Mimicking the one-dimensional marginal distributions of processes having an Itô differential », 1986.
* J. Guyon, P. Henry-Labordère, « Being particular about calibration », *Risk*, 2012 ; *Nonlinear Option Pricing*, CRC, 2013.
* C. Bayer, P. Friz, J. Gatheral, « Pricing under rough volatility », *Quant. Finance*, 2016 ; J. Gatheral, T. Jaisson, M. Rosenbaum, « Volatility is rough », *Quant. Finance*, 2018.
* M. Bennedsen, A. Lunde, M. Pakkanen, « Hybrid scheme for Brownian semistationary processes », *Finance & Stochastics*, 2017 ; R. McCrickerd, M. Pakkanen, « Turbocharging Monte Carlo pricing for the rough Bergomi model », 2018.
* P. Hagan, D. Kumar, A. Lesniewski, D. Woodward, « Managing smile risk », *Wilmott*, 2002.
* F. Longstaff, E. Schwartz, « Valuing American options by simulation: a simple least-squares approach », *RFS*, 2001.
* L. Andersen, M. Broadie, « Primal-dual simulation algorithm for pricing multidimensional American options », *Management Science*, 2004 ; L.C.G. Rogers, « Monte Carlo valuation of American options », *Math. Finance*, 2002.
* M. Broadie, P. Glasserman, S. Kou, « A continuity correction for discrete barrier options », *Math. Finance*, 1997.
* A. Kemna, A. Vorst, « A pricing method for options based on average asset values », *J. Banking & Finance*, 1990.
* M. Giles, « Multilevel Monte Carlo path simulation », *Operations Research*, 2008.
* M. Giles, P. Glasserman, « Smoking adjoints: fast Monte Carlo Greeks », *Risk*, 2006 ; L. Capriotti, « Fast Greeks by algorithmic differentiation », *J. Comp. Finance*, 2011.
* P. Glasserman, P. Heidelberger, P. Shahabuddin, « Asymptotically optimal importance sampling and stratification for pricing path-dependent options », *Math. Finance*, 1999.
* M. Broadie, P. Glasserman, « Estimating security price derivatives using simulation », *Management Science*, 1996.
* S. Joe, F. Kuo, « Constructing Sobol sequences with better two-dimensional projections », *SIAM J. Sci. Comput.*, 2008 ; A. Owen, « Scrambled net variance for integrals of smooth functions », *Ann. Stat.*, 1997.
* D. Brigo, F. Mercurio, *Interest Rate Models — Theory and Practice*, Springer, 2006.
* L. Andersen, V. Piterbarg, *Interest Rate Modeling*, 2010.
* F. Jamshidian, « An exact bond option formula », *J. Finance*, 1989.
* C. Hunter, P. Jäckel, M. Joshi, « Getting the drift », *Risk*, 2001.
* A. Lyashenko, F. Mercurio, « Looking forward to backward-looking rates », 2019.
* J. Gregory, *The xVA Challenge*, Wiley, 4e éd., 2020 ; M. Pykhtin, S. Zhu, « A guide to modelling counterparty credit risk », *GARP*, 2007.
* Comité de Bâle, *Minimum capital requirements for market risk* (FRTB), 2019.
