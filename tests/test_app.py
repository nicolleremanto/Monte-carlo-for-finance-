"""Tests du pont Python de l'interface web (app/bridge.py) et du manifeste Pyodide."""

import json
import sys
from itertools import pairwise
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import bridge  # noqa: E402
import make_manifest  # noqa: E402


def call(name, **args):
    out = json.loads(bridge.call(name, json.dumps(args)))
    assert out["ok"], out.get("error")
    return out["result"]


def test_manifest_is_in_sync():
    """app/manifest.json doit lister exactement les modules de mcfin (chargés par Pyodide)."""
    on_disk = json.loads((ROOT / "app" / "manifest.json").read_text())
    assert on_disk == make_manifest.build(), "lancer : python app/make_manifest.py"


def test_bs_overview_methods_agree():
    r = call(
        "bs_overview", S0=100, K=105, T=1, r=0.03, q=0.01, sigma=0.25, option_type="call", n_paths=20_000
    )
    for row in r["compare"]:
        tol = 4 * row["err_std"] if row["err_std"] else 5e-4
        assert abs(row["prix"] - r["price"]) < tol, row["méthode"]
    assert len(r["greeks"]) == 16 and len(r["curve"]["S"]) == 121


def test_bs_american_and_boundary():
    r = call("bs_american", S0=36, K=40, T=1, r=0.06, sigma=0.2, option_type="put", n_paths=20_000)
    assert abs(r["american_pde"] - r["american_tree"]) < 2e-3
    assert r["premium"] > 0.5
    s = r["boundary"]["S"]
    assert max(s) < 40 and all(b >= a - 1e-9 for a, b in pairwise(s))  # croissante en t


def test_bs_hedging_payload():
    r = call(
        "bs_hedging",
        S0=100,
        K=100,
        T=1,
        r=0.03,
        sigma=0.25,
        sigma_real=0.15,
        n_rebalancing=26,
        cost=0.004,
        n_paths=2_000,
    )
    assert r["mean"] > 0 and r["scatter"] is not None and r["leland"] is not None
    assert sum(r["hist"]["counts"]) == 2_000


@pytest.mark.parametrize("model", ["bs", "heston", "merton", "rbergomi", "localvol"])
@pytest.mark.parametrize("product", ["european", "digital", "asian", "barrier", "lookback"])
def test_mc_price_every_model_and_product(model, product):
    r = call(
        "mc_price",
        model=model,
        product=product,
        n_paths=4_000,
        spot=100,
        strike=100,
        maturity=0.5,
        barrier=80,
        control_variate=True,
    )
    lo, hi = r["ci"]
    assert lo < r["price"] < hi and r["stderr"] > 0
    assert len(r["paths"]["S"]) == 30
    if r["reference"] is not None:
        assert abs(r["z"]) < 4


def test_mc_convergence_slopes():
    r = call("mc_convergence", model="bs", product="european", n_paths=16_384)
    slopes = {s["label"]: s["slope"] for s in r["series"]}
    assert -0.65 < slopes["Pseudo-aléatoire"] < -0.35
    assert slopes["Sobol + pont brownien"] < slopes["Pseudo-aléatoire"]


def test_errors_are_reported_not_raised():
    out = json.loads(bridge.call("mc_price", json.dumps({"n_paths": -1})))
    assert not out["ok"] and "n_paths" in out["error"]
    out = json.loads(bridge.call("inconnue", "{}"))
    assert not out["ok"]
