"""Convergence et réduction de variance sur une asiatique arithmétique (GBM).

Compare, à budget égal, l'erreur standard de :
pseudo-aléatoire, antithétiques, Sobol + pont brownien (QMC randomisé),
variable de contrôle (asiatique géométrique, Kemna-Vorst), stratification
de W(T) et échantillonnage préférentiel.
"""

import matplotlib.pyplot as plt
import numpy as np
from _style import INK2, MARKERS, SERIES, save, setup

import mcfin as mc
from mcfin.analytics import geometric_asian_price

setup()
model = mc.BlackScholes(spot=100, vol=0.25, rate=0.03)
fix = np.arange(1, 13) / 12
asian = mc.AsianOption(100, fix, "call")
geo = mc.AsianOption(100, fix, "call", "geometric")
cv = mc.ControlVariate(geo.payoff, geometric_asian_price(100, 100, 1.0, 0.03, 0.25, fix))

N = 2**16
rows = [
    ("Pseudo-aléatoire", mc.MonteCarloEngine(N, seed=1).price(model, asian)),
    ("Antithétiques", mc.MonteCarloEngine(N, seed=1, antithetic=True).price(model, asian)),
    ("Moment matching", mc.MonteCarloEngine(N, seed=1, moment_matching=True).price(model, asian)),
    (
        "Sobol + pont brownien",
        mc.MonteCarloEngine(N, seed=1, method="sobol", construction="bridge").price(model, asian),
    ),
    (
        "Stratification W(T) + pont",
        mc.MonteCarloEngine(N, seed=1, n_strata=256, batch_size=2**14, construction="bridge").price(
            model, asian
        ),
    ),
    ("Variable de contrôle géométrique", mc.MonteCarloEngine(N, seed=1).price(model, asian, [cv])),
]
base = rows[0][1].stderr
print(f"{'Méthode':<34}{'Prix':>10}{'Err. std':>11}{'Gain variance':>15}")
for name, r in rows:
    print(f"{name:<34}{r.price:>10.5f}{r.stderr:>11.5f}{(base / r.stderr) ** 2:>15.1f}")
print(
    "NB : l'erreur standard « naïve » du moment matching n'est pas fiable (tirages\n"
    "rendus dépendants par la renormalisation) ; son intérêt est surtout le biais réduit\n"
    "sur les grandeurs dont on impose les moments."
)

# convergence en fonction du nombre de trajectoires
ns = 2 ** np.arange(10, 19)
curves = {
    "Pseudo-aléatoire": {},
    "Antithétiques": dict(antithetic=True),
    "Sobol + pont brownien": dict(method="sobol", construction="bridge"),
}
fig, ax = plt.subplots(figsize=(7, 4.2))
for (label, kw), color, mk in zip(curves.items(), SERIES, MARKERS, strict=False):
    err = [mc.MonteCarloEngine(int(n), seed=3, **kw).price(model, asian).stderr for n in ns]
    ax.loglog(ns, err, color=color, marker=mk, label=label)
    ax.annotate(
        label,
        (ns[-1], err[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        color=INK2,
        fontsize=8,
        va="center",
    )
ax.loglog(ns, 2.0 * rows[0][1].stderr * np.sqrt(N / ns), color="#898781", lw=1, ls="--")
ax.annotate(
    "pente −1/2",
    (ns[2], 2.0 * base * np.sqrt(N / ns[2])),
    color="#898781",
    fontsize=8,
    xytext=(4, 4),
    textcoords="offset points",
)
ax.set_xlabel("Nombre de trajectoires N")
ax.set_ylabel("Erreur standard du prix")
ax.set_title("Asiatique arithmétique : convergence MC vs QMC")
ax.set_xlim(ns[0] / 1.3, ns[-1] * 6)
ax.legend(loc="lower left")
save(fig, "convergence_qmc.png")
