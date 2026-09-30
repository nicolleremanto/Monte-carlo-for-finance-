"""Greeks Monte Carlo : bump & revalue (CRN), ratio de vraisemblance, AAD.

1. Asiatique sous GBM : les trois méthodes sur delta et vega.
2. Digitale : le pathwise/AAD échoue (dérivée nulle p.s.), le LRM fonctionne.
3. Heston : 7 sensibilités en UNE passe adjointe vs 14 revalorisations.
"""

import time

import numpy as np

import mcfin as mc
from mcfin.analytics import bs_digital_price
from mcfin.greeks import aad, gbm_path_pricer, heston_pricer, likelihood_ratio_greeks

fix = np.arange(1, 13) / 12
model = mc.BlackScholes(spot=100, vol=0.25, rate=0.03)
asian = mc.AsianOption(100, fix, "call")
n = 200_000

t = time.perf_counter()
bump = mc.MonteCarloEngine(n, seed=1).greeks(model, asian, {"vol": 0.01}, second_order=False)
t_bump = time.perf_counter() - t
t = time.perf_counter()
lrm = likelihood_ratio_greeks(model, asian, n, seed=1)
t_lrm = time.perf_counter() - t
pricer = gbm_path_pricer(lambda s, xp: xp.maximum(sum(s) / len(s) - 100.0, 0.0), fix, 1.0)
z = np.random.default_rng(1).standard_normal((n, fix.size))
t = time.perf_counter()
ad = aad.aad_greeks(pricer, {"spot": 100.0, "vol": 0.25, "rate": 0.03}, z)
t_aad = time.perf_counter() - t
print("Asiatique arithmétique (12 fixings)")
print(f"{'':<10}{'delta':>10}{'vega':>10}{'temps (s)':>12}")
print(f"{'Bump CRN':<10}{bump['delta']:>10.4f}{bump['vol']:>10.3f}{t_bump:>12.2f}")
print(
    f"{'LRM':<10}{lrm['delta']:>10.4f}{lrm['vega']:>10.3f}{t_lrm:>12.2f}   "
    f"(err. std : {lrm['delta_stderr']:.4f} / {lrm['vega_stderr']:.3f})"
)
print(
    f"{'AAD':<10}{ad['spot']:>10.4f}{ad['vol']:>10.3f}{t_aad:>12.2f}   "
    f"(err. std : {ad['spot_stderr']:.4f} / {ad['vol_stderr']:.3f})"
)

dig_pricer = gbm_path_pricer(lambda s, xp: (s[-1].value > 105) * 1.0 + 0.0 * s[-1], [1.0], 1.0)
dig_smooth = gbm_path_pricer(lambda s, xp: xp.smooth_step(s[-1] - 105.0, 0.5), [1.0], 1.0)
z1 = np.random.default_rng(2).standard_normal((n, 1))
p = {"spot": 100.0, "vol": 0.25, "rate": 0.03}
lrm_d = likelihood_ratio_greeks(model, mc.DigitalOption(105, 1.0), n, seed=2)
print("\nDigitale K=105 : delta")
print(f"  AAD brut      : {aad.aad_greeks(dig_pricer, p, z1)['spot']:.5f}   (faux : dérivée nulle p.s.)")
print(f"  AAD lissé     : {aad.aad_greeks(dig_smooth, p, z1)['spot']:.5f}   (sigmoïde, biais O(ε²))")
print(f"  LRM           : {lrm_d['delta']:.5f} ± {lrm_d['delta_stderr']:.5f}")
h = 1e-4
up, dn = bs_digital_price(100 + h, 105, 1, 0.03, 0.25), bs_digital_price(100 - h, 105, 1, 0.03, 0.25)
print(f"  exact         : {(up - dn) / (2 * h):.5f}")

hp = heston_pricer(100.0, 1.0, 50)
params = dict(spot=100.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, rate=0.02)
zh = np.random.default_rng(3).standard_normal((50_000, 50, 2))
t = time.perf_counter()
g = aad.aad_greeks(hp, params, zh, batch_size=10_000)
t_aad = time.perf_counter() - t


def value(pp):
    tape = aad.Tape()
    return float(hp({k: tape.variable(v) for k, v in pp.items()}, zh).value)


t = time.perf_counter()
fd = {}
for k in params:
    h = 1e-4 * max(1.0, abs(params[k]))
    up, dn = dict(params), dict(params)
    up[k] += h
    dn[k] -= h
    fd[k] = (value(up) - value(dn)) / (2 * h)
t_fd = time.perf_counter() - t
print(f"\nHeston (Euler, 50 pas, 50k trajectoires) : prix {g['price']:.4f}")
print(f"{'param':<8}{'AAD':>12}{'diff. finies':>15}")
for k in params:
    print(f"{k:<8}{g[k]:>12.4f}{fd[k]:>15.4f}")
print(f"temps AAD : {t_aad:.2f}s (1 passe avant + 1 arrière) | bumps : {t_fd:.2f}s (14 pricings)")
