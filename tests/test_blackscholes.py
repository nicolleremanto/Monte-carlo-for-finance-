"""Tests du sous-package mcfin.blackscholes (simulation, EDP, couverture, inférence)."""
import numpy as np
import pytest
from scipy import stats
from scipy.special import ndtri

from mcfin.analytics import bs_greeks, leisen_reimer_price
from mcfin.blackscholes import (REJECTION_CONSTANT, BlackScholesPricer, box_muller,
                                bs_pde_price, convergence_rate, coverage_study, fit_gbm_mle,
                                inverse_normal_bsm, kamal_derman_std, leland_volatility,
                                marsaglia_polar, range_volatility, rejection_laplace,
                                simulate_delta_hedge, simulate_ohlc)

OPT = BlackScholesPricer(S0=100, K=105, T=1.0, r=0.03, sigma=0.25, q=0.01)


# --- génération de la loi normale --------------------------------------------------
def test_beasley_springer_moro_accuracy():
    u = np.concatenate([np.linspace(1e-10, 1 - 1e-10, 200_001), [1e-12, 0.5, 1 - 1e-12]])
    assert np.max(np.abs(inverse_normal_bsm(u) - ndtri(u))) < 5e-9


@pytest.mark.parametrize("sampler", ["box_muller", "polar", "rejection"])
def test_normal_samplers(sampler):
    rng = np.random.default_rng(42)
    n = 200_000
    if sampler == "box_muller":
        x, acc = box_muller(n, rng), 1.0
    elif sampler == "polar":
        x, acc = marsaglia_polar(n, rng)
        assert abs(acc - np.pi / 4) < 0.01
    else:
        x, acc = rejection_laplace(n, rng)
        assert abs(acc - 1 / REJECTION_CONSTANT) < 0.01
    assert x.size == n
    assert stats.kstest(x, "norm").pvalue > 0.001
    assert abs(x.mean()) < 4 / np.sqrt(n) and abs(x.var() - 1) < 4 * np.sqrt(2 / n)


# --- EDP ------------------------------------------------------------------------------
@pytest.mark.parametrize("otype", ["call", "put"])
def test_crank_nicolson_second_order(otype):
    ref = bs_greeks(100, 105, 1.0, 0.03, 0.25, 0.01, otype)
    errs = [abs(bs_pde_price(100, 105, 1.0, 0.03, 0.25, 0.01, otype, n_space=n, n_time=n // 2).price
                - float(ref["price"])) for n in (100, 200, 400)]
    assert errs[-1] < 2e-4
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(rates > 1.8)                      # ordre 2 en (Δx, Δt)
    res = bs_pde_price(100, 105, 1.0, 0.03, 0.25, 0.01, otype, n_space=400, n_time=200)
    for k in ("delta", "gamma", "theta"):
        assert abs(getattr(res, k) - float(ref[k])) < 1e-3 * max(1, abs(float(ref[k])))


def test_american_pde_matches_tree():
    tree = leisen_reimer_price(36, 40, 1, 0.06, 0.2, n_steps=4001)
    res = bs_pde_price(36, 40, 1, 0.06, 0.2, option_type="put", american=True, n_space=800, n_time=400)
    assert abs(res.price - tree) < 2e-4
    tau, s_star = res.exercise_boundary[:, 0], res.exercise_boundary[:, 1]
    assert np.all(s_star < 40) and np.all(np.diff(s_star) <= 1e-9)  # frontière décroissante en τ
    # call américain sans dividende = européen (Merton : jamais d'exercice anticipé)
    call = bs_pde_price(100, 100, 1, 0.05, 0.3, 0.0, "call", american=True, n_space=400, n_time=200)
    assert abs(call.price - OPT.__class__(100, 100, 1, 0.05, 0.3).price()) < 1e-3


# --- façade -----------------------------------------------------------------------------
def test_pricer_all_methods_agree():
    rows = {r["méthode"]: r for r in OPT.compare(n_paths=2**15)}
    ref = OPT.price()
    for name, r in rows.items():
        tol = 4 * r["err_std"] if r["err_std"] else 5e-4
        assert abs(r["prix"] - ref) < tol, name
    assert abs(OPT.implied_vol(ref) - 0.25) < 1e-10
    with pytest.raises(ValueError):
        BlackScholesPricer(100, 100, -1, 0.0, 0.2)


def test_control_variate_reduces_variance():
    plain = OPT.monte_carlo(50_000, seed=1)
    cv = OPT.monte_carlo(50_000, seed=1, control_variate=True)
    assert cv.stderr < plain.stderr / 2


# --- diagnostics statistiques ---------------------------------------------------------------
def test_mc_confidence_intervals_have_nominal_coverage():
    rep = coverage_study(lambda s: OPT.monte_carlo(2_000, seed=s), OPT.price(), 400)
    assert rep.passed, rep


def test_rqmc_is_unbiased_and_nearly_covered():
    rep = coverage_study(lambda s: OPT.monte_carlo(2**12, seed=s, method="sobol"), OPT.price(), 200)
    assert rep.bias_pvalue > 0.01 and rep.coverage > 0.85


def test_convergence_rates():
    mc = convergence_rate(lambda n: OPT.monte_carlo(n, seed=1), [2**k for k in range(10, 17)])
    lo, hi = mc["slope_ci"]
    assert lo < -0.5 < hi
    qmc = convergence_rate(lambda n: OPT.monte_carlo(n, seed=1, method="sobol"),
                           [2**k for k in range(10, 17)])
    assert qmc["slope"] < -0.8                        # ~ N^{-1} (à log près)


# --- couverture delta ---------------------------------------------------------------------------
def test_hedging_error_scales_like_kamal_derman():
    stds = []
    for n in (13, 52, 208):
        h = simulate_delta_hedge(100, 100, 1.0, 0.03, 0.2, 0.2, mu=0.12, n_rebalancing=n,
                                 n_paths=10_000, seed=n)
        assert abs(h.mean) < 4 * h.std / np.sqrt(10_000)          # la tendance μ n'a pas de prix
        assert abs(h.std / kamal_derman_std(100, 100, 1.0, 0.03, 0.2, n) - 1) < 0.1
        stds.append(h.std)
    slope = np.polyfit(np.log([13, 52, 208]), np.log(stds), 1)[0]
    assert abs(slope + 0.5) < 0.05


def test_volatility_misspecification_gamma_pnl():
    h = simulate_delta_hedge(100, 100, 1.0, 0.03, 0.15, 0.25, n_rebalancing=500, n_paths=4000)
    assert np.corrcoef(h.pnl, h.gamma_pnl)[0, 1] > 0.98
    assert abs(h.mean - h.gamma_pnl.mean()) < 0.05
    # couvrir à la vraie vol rend le P&L (quasi) déterministe
    h2 = simulate_delta_hedge(100, 100, 1.0, 0.03, 0.15, 0.25, sigma_hedge=0.15, n_rebalancing=500,
                              n_paths=4000)
    expected = (OPT.__class__(100, 100, 1, 0.03, 0.25).price()
                - OPT.__class__(100, 100, 1, 0.03, 0.15).price()) * np.exp(0.03)
    assert abs(h2.mean - expected) < 0.02 and h2.std < 0.25 * h.std


def test_leland_covers_transaction_costs():
    k, n = 0.004, 52
    kw = dict(S0=100, K=100, T=1.0, r=0.03, sigma_real=0.2, n_rebalancing=n, cost=k,
              setup_costs=False, n_paths=10_000, seed=3)
    bs = simulate_delta_hedge(sigma_implied=0.2, **kw)
    le = simulate_delta_hedge(sigma_implied=leland_volatility(0.2, k, 1 / n), **kw)
    assert bs.mean < -0.3
    assert abs(le.mean) < 4 * le.std / np.sqrt(10_000)


# --- inférence -------------------------------------------------------------------------------------
def test_mle_coverage_and_drift_blur():
    hits_s = hits_m = 0
    n_rep = 200
    for s in range(n_rep):
        prices = simulate_ohlc(100, 0.08, 0.2, 504, 5, seed=s)["close"]
        fit = fit_gbm_mle(np.concatenate([[100.0], prices]), 1 / 252)
        ci = fit.ci()
        hits_s += ci["sigma"][0] <= 0.2 <= ci["sigma"][1]
        hits_m += ci["mu"][0] <= 0.08 <= ci["mu"][1]
    assert 0.9 < hits_s / n_rep < 0.99 and 0.9 < hits_m / n_rep < 0.99
    # précision de μ en σ/√T : ~ 14 % sur 2 ans, quelle que soit la fréquence
    assert 0.12 < fit.se_mu < 0.16 and fit.se_sigma < 0.01


def _range_study(steps_per_day, n_rep=150):
    est = {m: [] for m in ("close_to_close", "parkinson", "garman_klass", "rogers_satchell")}
    for s in range(n_rep):
        o = simulate_ohlc(100, 0.0, 0.2, 21, steps_per_day, seed=s)
        for m in est:
            est[m].append(range_volatility(o, method=m) ** 2)
    return {m: np.asarray(v) for m, v in est.items()}


def test_range_estimators_efficiency_and_discretisation_bias():
    fine = _range_study(400)
    var_cc = np.var(fine["close_to_close"])
    assert var_cc / np.var(fine["parkinson"]) > 3          # théorie : ≈ 5,2
    assert var_cc / np.var(fine["garman_klass"]) > 4       # théorie : ≈ 7,4
    assert abs(fine["close_to_close"].mean() / 0.04 - 1) < 0.05
    # plus haut/bas relevés sur une grille discrète => amplitude sous-estimée :
    # biais négatif, qui diminue quand la grille intra-journalière s'affine
    coarse = _range_study(25)
    for m in ("parkinson", "garman_klass", "rogers_satchell"):
        b_coarse, b_fine = coarse[m].mean() / 0.04 - 1, fine[m].mean() / 0.04 - 1
        assert b_coarse < b_fine < 0.02, m
