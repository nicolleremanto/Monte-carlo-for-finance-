"""XVA d'un portefeuille de swaps sous Hull-White : profils d'exposition,
CVA / DVA / FVA, effet du netting et du collatéral (CSA, MPOR 10 jours)."""
import matplotlib.pyplot as plt
import numpy as np
from _style import MARKERS, SERIES, save, setup

from mcfin.market import NelsonSiegelSvensson
from mcfin.rates import HullWhite
from mcfin.risk import CSA, Counterparty, InterestRateSwap, compute_xva, simulate_exposure

setup()
curve = NelsonSiegelSvensson(0.025, -0.005, 0.01, 0.005, 2.0, 5.0)
hw = HullWhite(a=0.03, sigma=0.009, curve=curve)
par = lambda T: (1 - curve.df(T)) / np.sum(curve.df(np.arange(1.0, T + 0.01)))
book = [InterestRateSwap(10e6, par(10.0), 0.0, 10.0, 1.0, payer=True),
        InterestRateSwap(5e6, par(5.0), 0.0, 5.0, 1.0, payer=False),
        InterestRateSwap(8e6, par(7.0) + 0.002, 0.0, 7.0, 1.0, payer=True)]
times = np.arange(1 / 12, 10.0 + 1e-9, 1 / 12)
cpty, bank = Counterparty(cds_spread=0.012, recovery=0.4), Counterparty(cds_spread=0.006)

unc = simulate_exposure(hw, book, times, n_paths=20_000, seed=1)
col = simulate_exposure(hw, book, times, n_paths=20_000, seed=1,
                        csa=CSA(threshold_cpty=250_000, threshold_bank=250_000, mpor=10 / 252))
single = [simulate_exposure(hw, [tr], times, n_paths=20_000, seed=1) for tr in book]
x_unc, x_col = compute_xva(unc, cpty, bank, 0.008), compute_xva(col, cpty, bank, 0.008)
cva_standalone = sum(compute_xva(s, cpty)["CVA"] for s in single)
print(f"{'':<22}{'CVA':>12}{'DVA':>12}{'FCA':>12}{'FBA':>12}{'Total':>12}")
for name, x in (("Sans collatéral", x_unc), ("CSA (seuil 250k)", x_col)):
    print(f"{name:<22}" + "".join(f"{x[k]:>12,.0f}" for k in ("CVA", "DVA", "FCA", "FBA", "total")))
print(f"CVA sans netting (somme des trades) : {cva_standalone:,.0f}  -> bénéfice de netting "
      f"{1 - x_unc['CVA'] / cva_standalone:.0%}")
print(f"EPE 1 an : {unc.epe:,.0f} | EEPE (Bâle) : {unc.eepe:,.0f} | PFE 95 % max : {unc.pfe.max():,.0f}")

fig, ax = plt.subplots(figsize=(7.5, 4.2))
for y, label, color, mk in [(unc.pfe, "PFE 95 % sans collatéral", SERIES[1], MARKERS[1]),
                            (unc.ee, "EE sans collatéral", SERIES[0], MARKERS[0]),
                            (col.ee, "EE avec CSA (MPOR 10 j)", SERIES[2], MARKERS[2])]:
    ax.plot(times, y / 1e3, color=color, marker=mk, markevery=12, label=label)
ax.set_xlabel("Horizon (années)")
ax.set_ylabel("Exposition (milliers)")
ax.set_title("Profils d'exposition d'un netting set de swaps (Hull-White)")
ax.legend(loc="upper right")
save(fig, "xva_exposure.png")
