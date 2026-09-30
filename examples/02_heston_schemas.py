"""Biais de discrétisation de Heston : Euler full truncation vs QE-M d'Andersen.

Paramètres « stressés » (condition de Feller violée : 2κθ/ξ² < 1), où
Euler est notoirement biaisé. Référence : formule semi-fermée (Lewis).
"""

import dataclasses

import matplotlib.pyplot as plt
import numpy as np
from _style import INK2, MARKERS, SERIES, save, setup

import mcfin as mc
from mcfin.analytics import heston_price

setup()
model = mc.Heston(spot=100, v0=0.04, kappa=1.0, theta=0.04, xi=0.9, rho=-0.7, rate=0.02)
opt = mc.EuropeanOption(100, 2.0)
ref = heston_price(100, 100, 2.0, 0.02, 0.0, model.params)[0]
print(f"Feller 2κθ/ξ² = {model.params.feller:.2f} ; prix de référence = {ref:.4f}")
steps = np.array([1, 2, 4, 8, 16, 32, 64])
fig, ax = plt.subplots(figsize=(7, 4.2))
for (scheme, label), color, mk in zip(
    [("euler", "Euler full truncation"), ("qe", "QE-M (Andersen 2008)")], SERIES, MARKERS, strict=False
):
    errs, ses = [], []
    for n in steps:
        m = dataclasses.replace(model, scheme=scheme, dt=1.0 / n)
        r = mc.MonteCarloEngine(200_000, seed=5, antithetic=True).price(m, opt)
        errs.append(r.price - ref)
        ses.append(r.stderr)
        print(f"{label:<24} pas = 1/{n:<3} biais = {r.price - ref:+.4f} ± {r.stderr:.4f}")
    ax.errorbar(steps, errs, yerr=2 * np.array(ses), color=color, marker=mk, capsize=3, label=label)
    ax.annotate(
        label,
        (steps[0], errs[0]),
        xytext=(10, 12 if scheme == "qe" else 0),
        textcoords="offset points",
        color=INK2,
        fontsize=8,
        va="center",
    )
ax.axhline(0, color="#c3c2b7", lw=1)
ax.set_xscale("log", base=2)
ax.set_xlabel("Nombre de pas par an")
ax.set_ylabel("Prix MC − prix exact (± 2 err. std)")
ax.set_title("Heston, Feller violé : biais de discrétisation")
ax.legend(loc="upper right")
save(fig, "heston_bias.png")
