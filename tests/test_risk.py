import numpy as np

from mcfin.market import NelsonSiegelSvensson
from mcfin.rates import HullWhite
from mcfin.risk import (
    CSA,
    Counterparty,
    InterestRateSwap,
    OptionBook,
    OptionPosition,
    compute_xva,
    simulate_exposure,
    simulate_pnl,
    var_es,
)

CURVE = NelsonSiegelSvensson(0.03, -0.01, 0.01, 0.005, 2.0, 5.0)
HW = HullWhite(a=0.05, sigma=0.01, curve=CURVE)


def _par_swap():
    sched = np.arange(1.0, 10.01)
    par = (1 - CURVE.df(10.0)) / np.sum(CURVE.df(sched))
    return InterestRateSwap(1e6, par, 0.0, 10.0, 1.0, payer=True), sched, par


def test_expected_exposure_equals_swaption():
    swap, sched, par = _par_swap()
    times = np.arange(0.5, 10.01, 0.5)
    res = simulate_exposure(HW, [swap], times, n_paths=40_000, seed=1)
    for T in (3.0, 6.0):
        i = int(np.argmin(np.abs(times - T)))
        swpt = 1e6 * HW.swaption(T, sched[sched > T], par)
        assert abs(res.ee[i] * CURVE.df(T) / swpt - 1) < 0.02


def test_xva_collateral_and_signs():
    swap, _, _ = _par_swap()
    times = np.arange(0.25, 10.01, 0.25)
    cpty, bank = Counterparty(0.01), Counterparty(0.005)
    unc = compute_xva(simulate_exposure(HW, [swap], times, 20_000), cpty, bank, 0.005)
    col = compute_xva(simulate_exposure(HW, [swap], times, 20_000, csa=CSA()), cpty, bank, 0.005)
    assert unc["CVA"] > 0 and unc["DVA"] > 0
    assert col["CVA"] < 0.3 * unc["CVA"]
    # swap payeur + receveur identiques : netting parfait => exposition nulle
    rec = InterestRateSwap(swap.notional, swap.fixed_rate, 0.0, 10.0, 1.0, payer=False)
    flat = simulate_exposure(HW, [swap, rec], times, 5_000)
    assert np.allclose(flat.ee, 0.0, atol=1e-6)


def test_var_es():
    book = OptionBook(
        spots=np.array([100.0, 50.0]),
        vols=np.array([0.25, 0.35]),
        corr=np.array([[1, 0.5], [0.5, 1]]),
        rate=0.02,
        positions=[
            OptionPosition(0, -1000, 100, 0.5, "put", 0.25),
            OptionPosition(1, 2000, 55, 1.0, "call", 0.35),
        ],
    )
    full = var_es(simulate_pnl(book, method="full", dist="normal"))
    dg = var_es(simulate_pnl(book, method="delta-gamma", dist="normal"))
    t = var_es(simulate_pnl(book, method="full", dist="student", dof=3.5))
    assert full["ES"] > 0 and full["VaR"] > 0
    assert abs(dg["VaR"] / full["VaR"] - 1) < 0.1
    assert np.isclose(full["ES_contributions"].sum(), full["ES"])  # Euler additif
    lo, hi = full["VaR_CI95"]
    assert lo <= full["VaR"] <= hi
    assert t["ES"] / t["VaR"] > full["ES"] / full["VaR"]  # queues épaisses
