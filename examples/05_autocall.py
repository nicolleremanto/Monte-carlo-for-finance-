"""Autocall Phoenix worst-of sur 3 indices : prix, statistiques, sensibilités.

Structure : 5 ans, observations trimestrielles, rappel à 100 %, coupon 2 %
par trimestre avec mémoire si worst-of >= 70 %, protection du capital à 60 %
(observée à maturité). Le modèle de diffusion est Black-Scholes multi-actifs ;
remplacer par LocalVol/LSV pour un pricing de production.
"""
import dataclasses

import matplotlib.pyplot as plt
import numpy as np
from _style import SERIES, save, setup

import mcfin as mc

setup()
corr = np.array([[1.0, 0.75, 0.6], [0.75, 1.0, 0.65], [0.6, 0.65, 1.0]])
model = mc.BlackScholes(spot=np.full(3, 100.0), vol=np.array([0.18, 0.22, 0.25]), rate=0.03,
                        div=np.array([0.03, 0.025, 0.02]), corr=corr)
dates = np.arange(1, 21) / 4
ac = mc.PhoenixAutocall(dates, autocall_barrier=1.0, coupon_barrier=0.7, coupon=0.02,
                        protection_barrier=0.6, memory=True, initial_levels=np.full(3, 100.0))
eng = mc.MonteCarloEngine(200_000, seed=7)
res = eng.price(model, ac)
an = ac.analytics(eng.simulate(model, ac.observation_times))
print(f"Prix : {res.price * 100:.2f} % du nominal ± {res.stderr * 100:.2f} %")
print(f"Durée de vie espérée : {an['expected_life']:.2f} ans")
print(f"Probabilité de perte en capital : {an['prob_capital_loss'] * 100:.1f} %"
      f" (perte moyenne sachant perte : {an['expected_loss_given_loss'] * 100:.1f} %)")

# sensibilités (bump & revalue, nombres aléatoires communs)
eng_g = mc.MonteCarloEngine(100_000, seed=11)
g = eng_g.greeks(model, ac, {"vol": 0.01, "rate": 0.001})
up = dataclasses.replace(model, corr=np.clip(corr + 0.05 * (1 - np.eye(3)), -1, 1))
dn = dataclasses.replace(model, corr=np.clip(corr - 0.05 * (1 - np.eye(3)), -1, 1))
corr_sens = (eng_g.price(up, ac).price - eng_g.price(dn, ac).price) / 0.10
print(f"Delta (choc parallèle, % nominal / +1 % spot) : {g['delta'] * 1.0 * 100:+.3f} %")
print(f"Gamma (% nominal / (1 % spot)²)               : {g['gamma'] * 1.0 * 100:+.4f} %")
print(f"Vega parallèle (pour +1 pt de vol)            : {g['vol'] * 0.01 * 100:+.3f} %")
print(f"Sensibilité corrélation (pour +1 pt)          : {corr_sens * 0.01 * 100:+.3f} %")
vol_side = "vendeur" if g["vol"] < 0 else "acheteur"
corr_side = "acheteur" if corr_sens > 0 else "vendeur"
print(f"=> l'investisseur est {vol_side} de volatilité et {corr_side} de corrélation ;")
print(f"   la banque émettrice porte les positions inverses (vega {'longue' if g['vol'] < 0 else 'courte'},"
      f" corrélation {'courte' if corr_sens > 0 else 'longue'}).")

p = an["redemption_prob_by_date"]
fig, ax = plt.subplots(figsize=(7.5, 4.0))
ax.bar(dates, p * 100, width=0.18, color=SERIES[0])
ax.set_xlabel("Date d'observation (années)")
ax.set_ylabel("Probabilité de remboursement (%)")
ax.set_title("Autocall Phoenix : profil de remboursement")
ax.annotate(f"maturité (non rappelé) : {p[-1] * 100:.1f} %", (dates[-1], p[-1] * 100),
            xytext=(-12, -4), textcoords="offset points", fontsize=8, color="#52514e", ha="right")
ax.grid(axis="x", visible=False)
save(fig, "autocall_redemption.png")
