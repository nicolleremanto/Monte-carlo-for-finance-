"""Pont entre l'interface web et la librairie ``mcfin``.

Chaque fonction publique prend un dictionnaire de paramètres (issu du
formulaire) et renvoie un dictionnaire sérialisable en JSON. Le même module
est exécuté soit dans le navigateur (Pyodide, WebAssembly), soit par le
serveur local ``app/serve.py`` : l'interface ne sait pas lequel des deux
calcule.

Point d'entrée unique : ``call(nom, json_params) -> json_resultat``.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

import numpy as np

import mcfin as mc
from mcfin.analytics import (
    barrier_price,
    bs_digital_price,
    bs_price,
    geometric_asian_price,
    heston_price,
    lookback_floating_price,
    merton_price,
)
from mcfin.blackscholes import (
    BlackScholesPricer,
    kamal_derman_std,
    leland_volatility,
    simulate_delta_hedge,
)

MAX_PATHS = 400_000


# --------------------------------------------------------------------------- utilitaires
def _clean(x: Any) -> Any:
    """Convertit récursivement numpy -> types JSON (NaN/inf -> None)."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _num(p: dict, key: str, default: float, lo: float | None = None, hi: float | None = None) -> float:
    v = float(p.get(key, default))
    if not math.isfinite(v):
        raise ValueError(f"paramètre « {key} » invalide")
    if lo is not None and v < lo:
        raise ValueError(f"« {key} » doit être ≥ {lo}")
    if hi is not None and v > hi:
        raise ValueError(f"« {key} » doit être ≤ {hi}")
    return v


def _histogram(x: np.ndarray, bins: int = 50) -> dict:
    x = np.asarray(x, dtype=float)
    lo, hi = np.quantile(x, [0.001, 0.999])
    if hi <= lo:
        hi = lo + 1e-9
    counts, edges = np.histogram(np.clip(x, lo, hi), bins=bins, range=(lo, hi))
    return {"edges": edges, "counts": counts}


def _bs_inputs(p: dict) -> BlackScholesPricer:
    return BlackScholesPricer(
        S0=_num(p, "S0", 100, 1e-6),
        K=_num(p, "K", 100, 1e-6),
        T=_num(p, "T", 1, 1e-4, 30),
        r=_num(p, "r", 0.03, -0.2, 0.5),
        sigma=_num(p, "sigma", 0.2, 1e-4, 3),
        q=_num(p, "q", 0.0, -0.2, 0.5),
        option_type=str(p.get("option_type", "call")),
    )


# --------------------------------------------------------------------------- Black-Scholes
def bs_overview(p: dict) -> dict:
    """Prix, Greeks, comparaison des méthodes et profils en fonction du spot."""
    opt = _bs_inputs(p)
    n = int(_num(p, "n_paths", 50_000, 1_000, MAX_PATHS))
    rows = opt.compare(n_paths=n)
    S = np.linspace(0.4 * opt.K, 1.6 * opt.K, 121)
    g = mc.analytics.bs_greeks(S, opt.K, opt.T, opt.r, opt.sigma, opt.q, opt.option_type)
    w = 1.0 if opt.option_type == "call" else -1.0
    return {
        "price": opt.price(),
        "greeks": opt.greeks(),
        "compare": rows,
        "curve": {
            "S": S,
            "price": g["price"],
            "intrinsic": np.maximum(w * (S - opt.K), 0.0),
            "delta": g["delta"],
            "gamma": g["gamma"],
        },
    }


def bs_american(p: dict) -> dict:
    """Européenne vs américaine : EDP (Brennan-Schwartz), arbre, Longstaff-Schwartz."""
    opt = _bs_inputs(p)
    t0 = time.perf_counter()
    euro = opt.price()
    pde = opt.pde(american=True, n_space=400, n_time=200)
    tree = opt.tree(american=True, n_steps=1001)
    n = int(_num(p, "n_paths", 50_000, 1_000, MAX_PATHS)) // 2
    ex_times = opt.T * np.arange(1, 51) / 50
    product = mc.BermudanOption(opt.K, ex_times, opt.option_type)
    model = opt.model()
    lsm = mc.LongstaffSchwartz(degree=3).fit(
        mc.MonteCarloEngine(n, seed=1, antithetic=True).simulate(model, ex_times), product
    )
    lo = lsm.price(mc.MonteCarloEngine(n, seed=2).simulate(model, ex_times), product)
    boundary = None
    if pde.exercise_boundary is not None:
        tau, s_star = pde.exercise_boundary[:, 0], pde.exercise_boundary[:, 1]
        # on écarte les tout premiers pas (phase de lissage de Rannacher, τ < 2Δt) :
        # la frontière y est mal résolue par la grille
        keep = tau >= 2 * opt.T / 200
        tau, s_star = tau[keep], s_star[keep]
        order = np.argsort(opt.T - tau)
        boundary = {"t": (opt.T - tau)[order], "S": s_star[order]}
    return {
        "european": euro,
        "american_pde": pde.price,
        "american_tree": tree,
        "lsm": lo.price,
        "lsm_stderr": lo.stderr,
        "premium": pde.price - euro,
        "boundary": boundary,
        "elapsed": time.perf_counter() - t0,
    }


def bs_hedging(p: dict) -> dict:
    """Couverture delta discrète : distribution du P&L, loi en n^{-1/2}, P&L de gamma, Leland."""
    opt = _bs_inputs(p)
    s_real = _num(p, "sigma_real", opt.sigma, 1e-4, 3)
    mu = _num(p, "mu", 0.08, -1, 1)
    n_reb = int(_num(p, "n_rebalancing", 52, 1, 2000))
    cost = _num(p, "cost", 0.0, 0.0, 0.2)
    n_paths = int(_num(p, "n_paths", 10_000, 500, 100_000))
    common = dict(S0=opt.S0, K=opt.K, T=opt.T, r=opt.r, q=opt.q, mu=mu, option_type=opt.option_type)
    t0 = time.perf_counter()
    main = simulate_delta_hedge(
        sigma_real=s_real, sigma_implied=opt.sigma, n_rebalancing=n_reb, n_paths=n_paths, cost=cost, **common
    )
    ns = [4, 12, 26, 52, 104, 252]
    std_sim = [
        simulate_delta_hedge(
            sigma_real=s_real,
            sigma_implied=opt.sigma,
            n_rebalancing=k,
            n_paths=min(n_paths, 5_000),
            seed=k,
            **common,
        ).std
        for k in ns
    ]
    same_vol = abs(s_real - opt.sigma) < 1e-12
    theory = (
        [kamal_derman_std(opt.S0, opt.K, opt.T, opt.r, opt.sigma, k, opt.q) for k in ns] if same_vol else None
    )
    rng = np.random.default_rng(0)
    pick = rng.choice(main.pnl.size, size=min(700, main.pnl.size), replace=False)
    out = {
        "premium": main.premium,
        "mean": main.mean,
        "std": main.std,
        "mean_stderr": main.std / math.sqrt(main.pnl.size),
        "kamal_derman": kamal_derman_std(opt.S0, opt.K, opt.T, opt.r, opt.sigma, n_reb, opt.q)
        if same_vol
        else None,
        "gamma_pnl_mean": float(main.gamma_pnl.mean()),
        "costs_mean": float(main.costs.mean()),
        "hist": _histogram(main.pnl),
        "scaling": {"n": ns, "std": std_sim, "theory": theory},
        "scatter": None if same_vol else {"x": main.gamma_pnl[pick], "y": main.pnl[pick]},
        "leland": None,
    }
    if cost > 0:
        s_l = leland_volatility(opt.sigma, cost, opt.T / n_reb)
        kw = dict(sigma_real=opt.sigma, n_rebalancing=n_reb, cost=cost, setup_costs=False, n_paths=n_paths)
        out["leland"] = {
            "sigma_leland": s_l,
            "mean_bs": simulate_delta_hedge(sigma_implied=opt.sigma, **kw, **common).mean,
            "mean_leland": simulate_delta_hedge(sigma_implied=s_l, **kw, **common).mean,
        }
    out["elapsed"] = time.perf_counter() - t0
    return out


# --------------------------------------------------------------------------- moteur Monte Carlo
def _model(p: dict):
    kind = p.get("model", "bs")
    S0 = _num(p, "spot", 100, 1e-6)
    r = _num(p, "rate", 0.03, -0.2, 0.5)
    q = _num(p, "div", 0.0, -0.2, 0.5)
    if kind == "bs":
        return mc.BlackScholes(spot=S0, vol=_num(p, "vol", 0.2, 1e-4, 3), rate=r, div=q)
    if kind == "heston":
        return mc.Heston(
            spot=S0,
            v0=_num(p, "v0", 0.04, 1e-6, 4),
            kappa=_num(p, "kappa", 1.5, 1e-4, 50),
            theta=_num(p, "theta", 0.04, 1e-6, 4),
            xi=_num(p, "xi", 0.6, 1e-4, 5),
            rho=_num(p, "rho", -0.7, -0.999, 0.999),
            rate=r,
            div=q,
            scheme="qe",
            dt=1 / 32,
        )
    if kind == "merton":
        return mc.MertonJumpDiffusion(
            spot=S0,
            vol=_num(p, "vol", 0.2, 1e-4, 3),
            lam=_num(p, "lam", 0.5, 0, 20),
            mu_j=_num(p, "mu_j", -0.1, -2, 2),
            sigma_j=_num(p, "sigma_j", 0.15, 0, 2),
            rate=r,
            div=q,
        )
    if kind == "rbergomi":
        return mc.RoughBergomi(
            spot=S0,
            xi0=_num(p, "xi0", 0.04, 1e-6, 4),
            eta=_num(p, "eta", 1.9, 0, 5),
            hurst=_num(p, "hurst", 0.1, 0.01, 0.5),
            rho=_num(p, "rho", -0.9, -0.999, 0.999),
            rate=r,
            div=q,
            dt=1 / 100,
        )
    if kind == "localvol":
        surf = mc.SSVISurface(
            sigma0=_num(p, "ssvi_sigma0", 0.18, 0.01, 2),
            sigma_inf=_num(p, "ssvi_sigma_inf", 0.22, 0.01, 2),
            lam=1.0,
            rho=_num(p, "ssvi_rho", -0.6, -0.99, 0.99),
            eta=_num(p, "ssvi_eta", 1.0, 0.01, 2),
            gamma=0.4,
        )
        return mc.LocalVol(spot=S0, surface=surf, rate=r, div=q, dt=1 / 50)
    raise ValueError(f"modèle inconnu : {kind}")


def _product(p: dict):
    kind = p.get("product", "european")
    K = _num(p, "strike", 100, 1e-6)
    T = _num(p, "maturity", 1, 1e-3, 10)
    ot = str(p.get("option_type", "call"))
    if kind == "european":
        return mc.EuropeanOption(K, T, ot)
    if kind == "digital":
        return mc.DigitalOption(K, T, ot)
    if kind == "asian":
        n_fix = int(_num(p, "n_fixings", 12, 1, 365))
        return mc.AsianOption(K, T * np.arange(1, n_fix + 1) / n_fix, ot)
    if kind == "barrier":
        return mc.BarrierOption(
            K,
            _num(p, "barrier", 80, 1e-6),
            T,
            ot,
            str(p.get("barrier_type", "down-and-out")),
            n_monitoring=int(_num(p, "n_monitoring", 50, 2, 1000)),
            monitoring="continuous",
        )
    if kind == "lookback":
        vol = float(p.get("vol", 0.2)) if p.get("model") == "bs" else None
        return mc.LookbackOption(
            T, ot, n_monitoring=int(_num(p, "n_monitoring", 100, 2, 1000)), bgk_sigma=vol
        )
    raise ValueError(f"produit inconnu : {kind}")


def _reference(p: dict, model, product) -> tuple[float | None, str]:
    """Prix de référence (formule fermée ou semi-analytique) quand il existe."""
    kind, prod = p.get("model", "bs"), p.get("product", "european")
    ot = str(p.get("option_type", "call"))
    S0, r, q = model.spot, float(p.get("rate", 0.03)), float(p.get("div", 0.0))
    if kind == "bs":
        s = model.vol
        if prod == "european":
            return float(bs_price(S0, product.strike, product.maturity, r, s, q, ot)), "Black-Scholes"
        if prod == "digital":
            return float(bs_digital_price(S0, product.strike, product.maturity, r, s, q, ot)), "digitale BS"
        if prod == "barrier":
            ref = barrier_price(
                S0, product.strike, product.barrier, product.maturity, r, s, q, ot, product.barrier_type
            )
            return float(ref), "Reiner-Rubinstein (continue)"
        if prod == "lookback":
            return lookback_floating_price(S0, product.maturity, r, s, q, ot), "Goldman-Sosin-Gatto (≈, BGK)"
    if kind == "heston" and prod == "european":
        return float(
            heston_price(S0, product.strike, product.maturity, r, q, model.params, ot)[0]
        ), "Heston (Lewis)"
    if kind == "merton" and prod == "european":
        ref = merton_price(
            S0, product.strike, product.maturity, r, model.vol, model.lam, model.mu_j, model.sigma_j, q, ot
        )
        return float(ref), "série de Merton"
    return None, ""


def _control_variates(p: dict, model, product) -> tuple[list, str]:
    if not p.get("control_variate"):
        return [], ""
    if p.get("model") == "bs" and p.get("product") == "asian":
        geo = mc.AsianOption(product.strike, product.fixing_times, product.option_type, "geometric")
        ref = geometric_asian_price(
            model.spot,
            product.strike,
            product.maturity,
            float(p.get("rate", 0.03)),
            model.vol,
            product.fixing_times,
            float(p.get("div", 0.0)),
            product.option_type,
        )
        return [
            mc.ControlVariate(geo.payoff, ref, "asiatique géométrique")
        ], "asiatique géométrique (Kemna-Vorst)"
    T = float(np.max(product.observation_times))
    q = float(p.get("div", 0.0))
    fwd = mc.ControlVariate(
        lambda paths: paths.spot[:, -1] * paths.df(-1), model.spot * math.exp(-q * T), "forward"
    )
    return [fwd], "sous-jacent actualisé (martingale)"


def _engine(p: dict, n: int | None = None, method: str | None = None, antithetic: bool | None = None):
    method = method or str(p.get("method", "pseudo"))
    n = n or int(_num(p, "n_paths", 50_000, 1_000, MAX_PATHS))
    if method == "sobol":
        n = 2 ** round(math.log2(n))
    return mc.MonteCarloEngine(
        n_paths=n,
        seed=int(p.get("seed", 42)),
        method=method,
        antithetic=bool(p.get("antithetic")) if antithetic is None else antithetic,
        construction="bridge" if method == "sobol" else "standard",
    )


def mc_price(p: dict) -> dict:
    model, product = _model(p), _product(p)
    cvs, cv_label = _control_variates(p, model, product)
    res = _engine(p).price(model, product, cvs)
    ref, ref_label = _reference(p, model, product)
    lo, hi = res.ci(0.95)
    T = float(np.max(product.observation_times))
    demo = mc.MonteCarloEngine(30, seed=7).simulate(model, T * np.arange(1, 61) / 60)
    sample = mc.MonteCarloEngine(5_000, seed=11).simulate(model, product.observation_times)
    return {
        "price": res.price,
        "stderr": res.stderr,
        "ci": [lo, hi],
        "n_paths": res.n_paths,
        "elapsed": res.elapsed,
        "method": res.method,
        "reference": ref,
        "reference_label": ref_label,
        "z": None if ref is None or res.stderr == 0 else (res.price - ref) / res.stderr,
        "variance_reduction": res.extra.get("variance_reduction"),
        "cv_label": cv_label,
        "paths": {"t": demo.times, "S": demo.spot},
        "barrier": getattr(product, "barrier", None),
        "hist": _histogram(product.payoff(sample)),
    }


def mc_convergence(p: dict) -> dict:
    """Erreur standard en fonction de N : pseudo, antithétiques, Sobol + pont brownien."""
    model, product = _model(p), _product(p)
    n_max = int(_num(p, "n_paths", 50_000, 1_000, MAX_PATHS))
    ns = [2**k for k in range(9, int(math.log2(n_max)) + 1)]
    out: dict = {"n": ns, "series": []}
    for label, method, anti in (
        ("Pseudo-aléatoire", "pseudo", False),
        ("Antithétiques", "pseudo", True),
        ("Sobol + pont brownien", "sobol", False),
    ):
        se = [_engine(p, n, method, anti).price(model, product).stderr for n in ns]
        slope = float(np.polyfit(np.log(ns), np.log(se), 1)[0])
        out["series"].append({"label": label, "stderr": se, "slope": slope})
    return out


FUNCTIONS = {f.__name__: f for f in (bs_overview, bs_american, bs_hedging, mc_price, mc_convergence)}


def call(name: str, args_json: str) -> str:
    """Point d'entrée unique : renvoie {"ok": true, "result": ...} ou {"ok": false, "error": ...}."""
    try:
        if name not in FUNCTIONS:
            raise ValueError(f"fonction inconnue : {name}")
        t0 = time.perf_counter()
        result = FUNCTIONS[name](json.loads(args_json or "{}"))
        return json.dumps({"ok": True, "result": _clean(result), "wall": time.perf_counter() - t0})
    except Exception as exc:  # renvoyé à l'interface, jamais silencieux
        return json.dumps({"ok": False, "error": f"{type(exc).__name__} : {exc}"})
