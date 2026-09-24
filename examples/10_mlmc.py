"""Multilevel Monte Carlo (Giles 2008) : décroissance de la variance par niveau
et gain de coût, Euler vs Milstein sous GBM."""
import matplotlib.pyplot as plt
import numpy as np
from _style import MARKERS, SERIES, save, setup

from mcfin.analytics import bs_price
from mcfin.variance_reduction import gbm_level_sampler, mlmc

setup()
ref = bs_price(100, 100, 1, 0.05, 0.2)
fig, ax = plt.subplots(figsize=(7, 4.2))
for scheme, color, mk in zip(("euler", "milstein"), SERIES, MARKERS):
    sampler = gbm_level_sampler(100, 100, 1.0, 0.05, 0.2, scheme=scheme)
    rng = np.random.default_rng(0)
    levels = np.arange(1, 9)
    v = []
    for l in levels:
        s1, s2, _ = sampler(int(l), 20_000, rng)
        v.append(s2 / 20_000 - (s1 / 20_000) ** 2)
    beta = -np.polyfit(levels, np.log2(v), 1)[0]
    ax.semilogy(levels, v, color=color, marker=mk, label=f"{scheme.capitalize()} (β ≈ {beta:.1f})",
                base=2)
    for eps in (0.02, 0.005):
        res = mlmc(sampler, eps=eps, seed=1)
        print(f"{scheme:<9} ε={eps:<6} prix={res.price:.4f} (exact {ref:.4f})  L={res.n_levels}  "
              f"coût MLMC={res.cost:.2e}  coût MC standard≈{res.std_mc_cost:.2e}  "
              f"gain ×{res.std_mc_cost / res.cost:.0f}")
ax.set_xlabel("Niveau l (pas = T / 2^l)")
ax.set_ylabel("Var[P_l − P_(l−1)]")
ax.set_title("MLMC : décroissance de la variance des corrections")
ax.legend(loc="upper right")
save(fig, "mlmc_variance.png")
