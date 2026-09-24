"""Structure par terme du skew ATM : rough Bergomi vs Heston.

Empiriquement (indices actions), le skew ATM ψ(T) = |∂σ/∂ln K| décroît en
T^{H-1/2} avec H ≈ 0.1 : explosion aux maturités courtes. Les modèles
markoviens (Heston) donnent un skew plat à court terme (ψ -> constante).
Le rough Bergomi reproduit la loi puissance (pente log-log ≈ H - 1/2 = -0.4).
"""
import matplotlib.pyplot as plt
import numpy as np
from _style import INK2, MARKERS, SERIES, save, setup

import mcfin as mc
from mcfin.analytics import heston_implied_vol, HestonParams, implied_vol

setup()
rb = mc.RoughBergomi(spot=100, xi0=0.04, eta=1.9, hurst=0.1, rho=-0.9, dt=1 / 500)
hp = HestonParams(0.04, 2.0, 0.04, 0.8, -0.7)
mats = np.array([0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0])
dk = 0.02
eng = mc.MonteCarloEngine(200_000, seed=2, antithetic=True)


def skew_rb(T):
    ks = np.array([-dk, dk]) * np.sqrt(T)
    K = 100 * np.exp(ks)
    v = [implied_vol(eng.price(rb, mc.EuropeanOption(k, T, "call" if k >= 100 else "put")).price,
                     100, k, T, 0, 0, "call" if k >= 100 else "put") for k in K]
    return abs(v[1] - v[0]) / (ks[1] - ks[0])


def skew_heston(T):
    ks = np.array([-dk, dk]) * np.sqrt(T)
    v = heston_implied_vol(100, 100 * np.exp(ks), T, 0, 0, hp)
    return abs(v[1] - v[0]) / (ks[1] - ks[0])


rbs = np.array([skew_rb(T) for T in mats])
hs = np.array([skew_heston(T) for T in mats])
slope = np.polyfit(np.log(mats), np.log(rbs), 1)[0]
print("T        skew rBergomi   skew Heston")
for T, a, b in zip(mats, rbs, hs):
    print(f"{T:<8.2f} {a:>12.3f} {b:>13.3f}")
print(f"pente log-log rBergomi = {slope:.3f} (théorie H - 1/2 = -0.4)")

fig, ax = plt.subplots(figsize=(7, 4.2))
for y, label, color, mk, dy in [(rbs, "Rough Bergomi (H = 0.1)", SERIES[0], MARKERS[0], 10),
                                (hs, "Heston", SERIES[1], MARKERS[1], -12)]:
    ax.loglog(mats, y, color=color, marker=mk, label=label)
    ax.annotate(label, (mats[0], y[0]), xytext=(8, dy), textcoords="offset points",
                color=INK2, fontsize=8, va="center")
ax.set_xlabel("Maturité T (années)")
ax.set_ylabel("Skew ATM |∂σ/∂ln K|")
ax.set_title(f"Skew ATM : loi puissance rough (pente {slope:.2f}) vs Heston")
ax.set_xlim(mats[0] / 1.2, mats[-1] * 1.3)
ax.legend(loc="upper right")
save(fig, "rough_skew.png")
