"""VaR / Expected Shortfall Monte Carlo d'un book d'options (horizon 10 jours).

Revalorisation complète vs approximations delta et delta-gamma(-vega) ;
lois gaussienne vs Student-t (queues épaisses) ; contributions d'Euler.
"""
import matplotlib.pyplot as plt
import numpy as np
from _style import INK2, SERIES, save, setup

from mcfin.risk import OptionBook, OptionPosition, simulate_pnl, var_es

setup()
book = OptionBook(
    spots=np.array([100.0, 50.0, 200.0]), vols=np.array([0.22, 0.35, 0.18]),
    corr=np.array([[1, 0.5, 0.4], [0.5, 1, 0.45], [0.4, 0.45, 1]]), rate=0.02,
    positions=[OptionPosition(0, -2000, 95, 0.25, "put", 0.26),        # put court (gamma négatif)
               OptionPosition(1, 3000, 55, 1.0, "call", 0.35),
               OptionPosition(2, -600, None),                          # couverture en actions
               OptionPosition(2, 1500, 190, 0.5, "put", 0.21)])
names = ["Put court A", "Call B", "Action C", "Put C"]
print(f"{'Méthode':<28}{'VaR 99 %':>12}{'ES 97.5 %':>12}   IC95 ES")
results = {}
for dist in ("normal", "student"):
    for meth in ("full", "delta", "delta-gamma"):
        pnl = simulate_pnl(book, 10 / 252, 200_000, dist=dist, dof=4, vol_of_vol=1.0, method=meth, seed=3)
        r = var_es(pnl)
        results[(dist, meth)] = (pnl, r)
        lo, hi = r["ES_CI95"]
        print(f"{dist + ' / ' + meth:<28}{r['VaR']:>12,.0f}{r['ES']:>12,.0f}   [{lo:,.0f} ; {hi:,.0f}]")
r = results[("student", "full")][1]
print("\nContributions d'Euler à l'ES (Student, full reval) :")
for n, c in zip(names, r["ES_contributions"]):
    print(f"  {n:<12} {c:>12,.0f}  ({c / r['ES']:.0%})")

pnl = results[("student", "full")][0].sum(axis=1)
fig, ax = plt.subplots(figsize=(7.5, 4.0))
lo, hi = np.quantile(pnl / 1e3, [0.0005, 0.9995])
ax.hist(pnl / 1e3, bins=120, range=(lo, hi), color=SERIES[0], alpha=0.9)
for val, lab, dy, ls in ((r["VaR"], "VaR 99 %", 0.9, "--"), (r["ES"], "ES 97.5 %", 0.75, ":")):
    ax.axvline(-val / 1e3, color=INK2, lw=1, ls=ls)
    ax.annotate(f"{lab} = {val / 1e3:,.1f}k", (-val / 1e3, ax.get_ylim()[1] * dy), xytext=(-6, 0),
                textcoords="offset points", ha="right", fontsize=8, color=INK2)
ax.set_xlim(lo, hi)
ax.set_xlabel("P&L à 10 jours (milliers) — quantiles 0,05 % à 99,95 %")
ax.set_ylabel("Nombre de scénarios")
ax.set_title("Distribution du P&L (Student-t, revalorisation complète)")
ax.grid(axis="x", visible=False)
save(fig, "var_distribution.png")
