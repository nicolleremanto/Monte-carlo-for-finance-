"""Laboratoire Black-Scholes : une option, quatre méthodes, et leur contrôle statistique.

1. formule fermée vs Monte Carlo vs QMC vs EDP vs arbre ;
2. générateurs de gaussiennes from scratch (inversion, Box-Muller, polaire, rejet) ;
3. couverture empirique des intervalles de confiance (MC vs QMC randomisé) ;
4. convergence de l'EDP : effet du lissage du payoff et de Rannacher (figure).
"""
import time

import matplotlib.pyplot as plt
import numpy as np
from _style import INK2, MARKERS, SERIES, save, setup
from scipy import stats
from scipy.special import ndtri

from mcfin.blackscholes import (BlackScholesPricer, box_muller, bs_pde_price, coverage_study,
                                inverse_normal_bsm, marsaglia_polar, rejection_laplace)

setup()
opt = BlackScholesPricer(S0=100, K=105, T=1.0, r=0.03, sigma=0.25, q=0.01)
print("1) Call S0=100 K=105 T=1 r=3% q=1% σ=25%")
print(f"{'méthode':<30}{'prix':>11}{'écart':>11}{'err. std':>10}{'temps':>9}")
for row in opt.compare():
    se = f"{row['err_std']:.5f}" if row["err_std"] else "-"
    print(f"{row['méthode']:<30}{row['prix']:>11.5f}{row['écart']:>+11.1e}{se:>10}{row['temps_s']:>8.3f}s")

print("\n2) Loi normale « from scratch » (10^6 tirages)")
rng = np.random.default_rng(0)
n = 1_000_000
u = rng.random(n)
for name, fn in [("Inversion (Beasley-Springer-Moro)", lambda: (inverse_normal_bsm(u), 1.0)),
                 ("Box-Muller", lambda: (box_muller(n, rng), 1.0)),
                 ("Polaire de Marsaglia", lambda: marsaglia_polar(n, rng)),
                 ("Rejet depuis Laplace", lambda: rejection_laplace(n, rng))]:
    t0 = time.perf_counter()
    x, acc = fn()
    dt = time.perf_counter() - t0
    print(f"  {name:<34} acceptation {acc:.3f}  KS p = {stats.kstest(x, 'norm').pvalue:.2f}"
          f"  kurtosis {stats.kurtosis(x):+.3f}  {dt:.2f}s")
print(f"  erreur max de l'inversion BSM vs Φ⁻¹ exacte : {np.abs(inverse_normal_bsm(u) - ndtri(u)).max():.1e}")

print("\n3) Couverture empirique des IC à 95 % (300 répétitions)")
print("   MC  :", coverage_study(lambda s: opt.monte_carlo(4_000, seed=s), opt.price(), 300))
print("   RQMC:", coverage_study(lambda s: opt.monte_carlo(2**12, seed=s, method="sobol"),
                                 opt.price(), 300))
print("   -> RQMC : sans biais, mais IC légèrement trop étroit. Avec 16 brouillages de 256 points,\n"
      "      l'estimateur par brouillage n'est pas encore gaussien pour un payoff à coin (queues\n"
      "      épaisses) : phénomène connu (L'Ecuyer, Munger & Tuffin 2010). Plus de brouillages\n"
      "      restaure la couverture mais dégrade la précision : c'est un arbitrage.")

ns = np.array([25, 50, 100, 200, 400, 800])
ref = opt.price()
configs = [("Crank-Nicolson brut", dict(rannacher_steps=0, smooth_payoff=False)),
           ("Euler implicite (θ = 1)", dict(theta=1.0, rannacher_steps=0)),
           ("CN + Rannacher + payoff lissé", dict())]
fig, ax = plt.subplots(figsize=(7, 4.2))
print("\n4) Erreur de l'EDP (grille N x N/2)")
for (label, kw), color, mk in zip(configs, SERIES, MARKERS):
    err = [abs(bs_pde_price(100, 105, 1.0, 0.03, 0.25, 0.01, "call", n_space=int(k),
                            n_time=int(k) // 2, **kw).price - ref) for k in ns]
    slope = np.polyfit(np.log(ns[2:]), np.log(err[2:]), 1)[0]
    print(f"  {label:<32} erreurs {' '.join(f'{e:.1e}' for e in err)}  ordre ≈ {-slope:.2f}")
    ax.loglog(ns, err, color=color, marker=mk, label=f"{label} (ordre {-slope:.1f})")
ax.set_xticks(ns, [str(k) for k in ns])
ax.minorticks_off()
ax.set_xlabel("Nombre de points d'espace N")
ax.set_ylabel("|prix EDP − formule fermée|")
ax.set_title("EDP de Black-Scholes : convergence des θ-schémas")
ax.legend(loc="lower left")
ax.annotate("oscillations : coin du payoff\nentre deux nœuds", (ns[3], 1e-3), color=INK2, fontsize=8)
save(fig, "bs_pde_convergence.png")
