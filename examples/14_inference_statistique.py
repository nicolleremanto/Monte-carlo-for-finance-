"""Inférence sur Black-Scholes : ce que les données permettent (ou non) d'estimer.

Précision de l'EMV en fonction de la fréquence d'échantillonnage, à horizon
fixé (5 ans) : σ s'estime d'autant mieux qu'on observe souvent, μ jamais
mieux que σ/sqrt(T). Puis efficacité des estimateurs de volatilité par
l'amplitude (Parkinson, Garman-Klass, Rogers-Satchell).
"""
import matplotlib.pyplot as plt
import numpy as np
from _style import INK2, MARKERS, SERIES, save, setup

from mcfin.blackscholes import fit_gbm_mle, range_volatility, simulate_ohlc

setup()
mu, sigma, years = 0.08, 0.20, 5
freqs = {"mensuelle": 12, "hebdomadaire": 52, "quotidienne": 252, "horaire": 252 * 8}
half_mu, half_sig = [], []
print(f"EMV sur {years} ans (μ = 8 %, σ = 20 %) : demi-largeur de l'IC à 95 %")
for name, f in freqs.items():
    rng = np.random.default_rng(1)
    n = years * f
    r = (mu - 0.5 * sigma**2) / f + sigma / np.sqrt(f) * rng.standard_normal(n)
    fit = fit_gbm_mle(100 * np.exp(np.concatenate([[0.0], np.cumsum(r)])), 1 / f)
    half_mu.append(1.96 * fit.se_mu)
    half_sig.append(1.96 * fit.se_sigma)
    print(f"  {name:<13} n = {n:>6}  μ̂ = {fit.mu:+.3f} ± {half_mu[-1]:.3f}   "
          f"σ̂ = {fit.sigma:.4f} ± {half_sig[-1]:.4f}")
x = np.array(list(freqs.values())) * years
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.loglog(x, np.array(half_mu) * 100, color=SERIES[0], marker=MARKERS[0], label="tendance μ")
ax.loglog(x, np.array(half_sig) * 100, color=SERIES[1], marker=MARKERS[1], label="volatilité σ")
for xi, lab in zip(x, freqs):
    ax.annotate(lab, (xi, half_sig[list(freqs).index(lab)] * 100), xytext=(0, -14),
                textcoords="offset points", ha="right" if lab == "horaire" else "center",
                fontsize=8, color=INK2)
ax.set_xlabel("Nombre d'observations sur 5 ans")
ax.set_ylabel("Demi-largeur de l'IC à 95 % (points de %)")
ax.set_title("Échantillonner plus souvent : σ se précise, μ jamais")
ax.legend(loc="center right")
save(fig, "bs_mle_precision.png")

print("\nVolatilité sur un mois (21 séances, σ = 20 %, 300 répétitions)")
est = {m: [] for m in ("close_to_close", "parkinson", "garman_klass", "rogers_satchell")}
for s in range(300):
    o = simulate_ohlc(100, 0.0, sigma, 21, 390, seed=s)
    for m in est:
        est[m].append(range_volatility(o, method=m))
base = np.var(np.square(est["close_to_close"]))
for m, v in est.items():
    v = np.asarray(v)
    print(f"  {m:<16} moyenne {v.mean():.4f}  écart-type {v.std():.4f}  "
          f"efficacité {base / np.var(v**2):.1f}")
