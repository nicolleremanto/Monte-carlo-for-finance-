"""Bermudéennes : borne inférieure Longstaff-Schwartz et borne duale Andersen-Broadie."""
import numpy as np

import mcfin as mc
from mcfin.analytics import leisen_reimer_price


def run(model, product, n=100_000, n_outer=500, n_inner=1000):
    lsm = mc.LongstaffSchwartz(degree=3).fit(
        mc.MonteCarloEngine(n, seed=1, antithetic=True).simulate(model, product.observation_times), product)
    lo = lsm.price(mc.MonteCarloEngine(n, seed=2).simulate(model, product.observation_times), product)
    up = mc.andersen_broadie_upper_bound(model, product, lsm, lo.price, n_outer, n_inner)
    return lo, up


put = mc.BermudanOption(40, np.arange(1, 11) / 10, "put")
lo, up = run(mc.BlackScholes(spot=36, vol=0.2, rate=0.06), put, n_outer=300, n_inner=500)
print("Put bermudéen (10 dates) S=36 K=40 σ=20% r=6% T=1")
print(f"  LSM (borne inf.)   : {lo.price:.4f} ± {lo.stderr:.4f}")
print(f"  A-B (borne sup.)   : {up.price:.4f} ± {up.stderr:.4f}  (écart de dualité {up.extra['duality_gap']:.4f})")
print(f"  Américain (arbre LR, 1001 pas) : {leisen_reimer_price(36, 40, 1, 0.06, 0.2):.4f}")

mc2 = mc.BlackScholes(spot=np.array([100.0, 100.0]), vol=0.2, rate=0.05, div=0.1, corr=np.eye(2))
maxcall = mc.BermudanOption(100, np.arange(1, 10) / 3, "call", "max")
lo, up = run(mc2, maxcall, n=200_000)
print("\nMax-call 2 actifs (Andersen-Broadie 2004, intervalle publié [13.892, 13.934])")
print(f"  LSM : {lo.price:.4f} ± {lo.stderr:.4f}  |  A-B : {up.price:.4f} ± {up.stderr:.4f}")
