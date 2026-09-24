"""Taux : Hull-White (exact) et LMM (mesure spot) — caplets, swaptions, bermudéennes."""
import numpy as np

from mcfin.market import NelsonSiegelSvensson
from mcfin.rates import HullWhite, LiborMarketModel, bermudan_swaption_hw, bermudan_swaption_lmm

curve = NelsonSiegelSvensson(0.03, -0.01, 0.01, 0.005, 2.0, 5.0)
hw = HullWhite(a=0.05, sigma=0.01, curve=curve)
pay = np.arange(6.0, 11.0)
K = float((curve.df(5.0) - curve.df(10.0)) / np.sum(curve.df(pay)))
print(f"Swap forward 5y5y : {K:.4%}")
print(f"Swaption payeuse 5y5y ATM (Jamshidian) : {hw.swaption(5.0, pay, K) * 1e4:.1f} bp de nominal")
b = bermudan_swaption_hw(hw, np.arange(1.0, 10.0), 10.0, K, n_paths=100_000)
print(f"Bermudéenne 1y..9y into 10y (LSM) : {b.price * 1e4:.1f} bp ± {b.stderr * 1e4:.1f}"
      f"  | max européennes co-terminales : {b.extra['max_european'] * 1e4:.1f} bp"
      f"  -> prime de switch {100 * (b.price / b.extra['max_european'] - 1):.0f} %")

lmm = LiborMarketModel(np.arange(0, 10.5, 0.5), curve, n_factors=3, steps_per_period=2)
paths = lmm.simulate(50_000, seed=1, antithetic=True)
print("\nLMM 3 facteurs, forwards 6M, vol abcd, corrélation exponentielle")
for i in (4, 10, 18):
    k = lmm.L0[i]
    mc = lmm.tau[i] * np.maximum(paths.forwards[:, i, i] - k, 0) / paths.numeraire[:, i + 1]
    print(f"  caplet {lmm.tenor[i]:>4.1f}y ATM : MC {mc.mean() * 1e4:.3f} bp ± {mc.std() / np.sqrt(mc.size) * 1e4:.3f}"
          f"  | Black {lmm.caplet_price(i, k) * 1e4:.3f} bp  (vol {lmm.caplet_black_vol(i):.2%})")
S, A = paths.swap(4, 20)
mc = A * np.maximum(S - S.mean(), 0) / paths.numeraire[:, 4]
print(f"  swaption 2y8y ATM : MC {mc.mean() * 1e4:.1f} bp | Rebonato (plein rang) "
      f"{lmm.swaption_rebonato(4, 20, S.mean()) * 1e4:.1f} bp  (3 facteurs => corrélations effectives\n"
      "  plus fortes => swaption plus chère que la formule de plein rang)")
bl = bermudan_swaption_lmm(lmm, 4, 20, float(S.mean()), n_paths=40_000)
print(f"  bermudéenne 2y..9.5y into 10y : {bl.price * 1e4:.1f} bp ± {bl.stderr * 1e4:.1f}")
