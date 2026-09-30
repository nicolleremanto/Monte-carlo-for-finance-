"""Couverture delta discrète : ce que la réplication de Black-Scholes donne en pratique.

1. erreur de couverture vs fréquence de rebalancement (loi en n^{-1/2}, Kamal-Derman) ;
2. volatilité mal spécifiée : P&L = ½∫ e^{r(T-t)} Γ S² (σ_i² - σ_r²) dt ;
3. coûts de transaction : volatilité de Leland.
"""
import matplotlib.pyplot as plt
import numpy as np
from _style import INK2, MARKERS, SERIES, save, setup

from mcfin.blackscholes import kamal_derman_std, leland_volatility, simulate_delta_hedge

setup()
S0, K, T, r = 100.0, 100.0, 1.0, 0.03
ns = np.array([4, 12, 26, 52, 104, 252, 504])
emp = []
print("1) Vendeur couvert d'un call ATM, σ réelle = σ implicite = 20 %, μ = 12 %")
print(f"{'n':>5}{'moyenne':>10}{'écart-type':>12}{'Kamal-Derman':>14}")
for n in ns:
    h = simulate_delta_hedge(S0, K, T, r, 0.2, 0.2, mu=0.12, n_rebalancing=int(n), n_paths=20_000,
                             seed=int(n))
    emp.append(h.std)
    print(f"{n:>5}{h.mean:>10.3f}{h.std:>12.3f}{kamal_derman_std(S0, K, T, r, 0.2, int(n)):>14.3f}")
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.loglog(ns, emp, color=SERIES[0], marker=MARKERS[0], label="Simulation (20 000 trajectoires)")
ax.loglog(ns, [kamal_derman_std(S0, K, T, r, 0.2, int(n)) for n in ns], color=SERIES[1],
          ls="--", label="Kamal-Derman : sqrt(π/4)·vega·σ/sqrt(n)")
ax.set_xlabel("Nombre de rebalancements sur 1 an")
ax.set_ylabel("Écart-type du P&L de couverture")
ax.set_title("La réplication n'est parfaite qu'en temps continu")
ax.legend(loc="upper right")
ax.annotate("pente −1/2", (ns[3], emp[3]), xytext=(10, 10), textcoords="offset points",
            color=INK2, fontsize=8)
save(fig, "bs_hedging_error.png")

print("\n2) Vol vendue 25 %, réalisée 15 %, couverture quotidienne")
h = simulate_delta_hedge(S0, K, T, r, 0.15, 0.25, n_rebalancing=252, n_paths=10_000)
print(f"   couverture à σ_implicite : P&L moyen {h.mean:.3f}, écart-type {h.std:.3f}, "
      f"corr(P&L, P&L de gamma) = {np.corrcoef(h.pnl, h.gamma_pnl)[0, 1]:.3f}")
h2 = simulate_delta_hedge(S0, K, T, r, 0.15, 0.25, sigma_hedge=0.15, n_rebalancing=252,
                          n_paths=10_000)
print(f"   couverture à σ_réalisée  : P&L moyen {h2.mean:.3f}, écart-type {h2.std:.3f} "
      "(verrouille (V(σ_i) - V(σ_r))e^{rT})")
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.scatter(h.gamma_pnl[:3000], h.pnl[:3000], s=6, color=SERIES[0], alpha=0.5)
lim = [h.gamma_pnl.min(), h.gamma_pnl.max()]
ax.plot(lim, lim, color=INK2, lw=1, ls="--")
ax.set_xlabel(r"$\frac{1}{2}\int_0^T e^{r(T-t)}\,\Gamma_t S_t^2\,(\sigma_i^2-\sigma_r^2)\,dt$  (théorie)")
ax.set_ylabel("P&L réalisé de la couverture")
ax.set_title("Vol mal spécifiée : le P&L est le « P&L de gamma »")
save(fig, "bs_gamma_pnl.png")

print("\n3) Coûts de transaction k = 0,4 % aller-retour, rebalancement hebdomadaire")
k, n = 0.004, 52
sL = leland_volatility(0.2, k, T / n)
kw = dict(S0=S0, K=K, T=T, r=r, sigma_real=0.2, n_rebalancing=n, cost=k, setup_costs=False,
          n_paths=20_000)
bs = simulate_delta_hedge(sigma_implied=0.2, **kw)
le = simulate_delta_hedge(sigma_implied=sL, **kw)
print(f"   prime et delta Black-Scholes (σ = 20 %)   : P&L moyen {bs.mean:+.3f}")
print(f"   prime et delta de Leland (σ_L = {sL:.2%}) : P&L moyen {le.mean:+.3f}")
