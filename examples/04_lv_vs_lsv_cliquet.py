"""Volatilité locale vs LSV : mêmes vanilles, exotiques différents.

LV et LSV (calibré par méthode particulaire) repricent la même nappe SSVI.
Mais la dynamique du smile diffère : la LV produit un smile *forward* trop
plat, ce qui sous-évalue les produits sensibles à la volatilité future
(cliquets, forward-starts). C'est l'argument central pour le LSV sur les
desks d'exotiques actions.
"""

import numpy as np

import mcfin as mc
from mcfin.analytics import implied_vol

surf = mc.SSVISurface(sigma0=0.18, sigma_inf=0.22, lam=1.0, rho=-0.6, eta=1.2, gamma=0.4)
S0, r, q, T = 100.0, 0.02, 0.0, 2.0
lv = mc.LocalVol(spot=S0, surface=surf, rate=r, div=q, dt=1 / 100)
lsv = mc.LocalStochasticVol(
    spot=S0, surface=surf, rate=r, div=q, v0=0.04, kappa=1.5, theta=0.04, xi=0.35, rho=-0.6, dt=1 / 100
).calibrate(T, n_particles=50_000)
eng = mc.MonteCarloEngine(100_000, seed=1, antithetic=True)

print("Reprice des vanilles (vol implicite, T = 1 an)")
print(f"{'K':>6}{'marché':>10}{'LV':>10}{'LSV':>10}")
F = S0 * np.exp(r)
for K in (80.0, 90.0, 100.0, 110.0, 120.0):
    ot = "call" if K >= F else "put"
    ivs = [
        implied_vol(eng.price(m, mc.EuropeanOption(K, 1.0, ot)).price, S0, K, 1.0, r, q, ot)
        for m in (lv, lsv)
    ]
    print(f"{K:>6.0f}{float(surf.implied_vol(np.log(K / F), 1.0)):>10.4f}{ivs[0]:>10.4f}{ivs[1]:>10.4f}")

reset = np.arange(1, 25) / 12
cliquet = mc.Cliquet(reset, local_floor=-0.02, local_cap=0.02, global_floor=0.0)
print("\nCliquet 2 ans, resets mensuels, cap/floor locaux ±2 %, floor global 0")
for name, m in (("LV", lv), ("LSV", lsv)):
    res = eng.price(m, cliquet)
    print(f"  {name:<4} {res.price * 100:.3f} % ± {res.stderr * 100:.3f} %")
