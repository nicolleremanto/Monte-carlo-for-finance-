"""Tests de propriétés (hypothesis) : relations d'absence d'arbitrage que toute
formule de Black-Scholes doit vérifier, sur des paramètres tirés au hasard."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from mcfin.analytics import bs_greeks, bs_price, implied_vol

spot = st.floats(50, 150)
strike = st.floats(40, 200)
mat = st.floats(0.05, 5.0)
rate = st.floats(-0.02, 0.08)
div = st.floats(0.0, 0.06)
vol = st.floats(0.05, 1.0)
SETTINGS = settings(max_examples=300, deadline=None)


@SETTINGS
@given(spot, strike, mat, rate, vol, div)
def test_put_call_parity(S, K, T, r, s, q):
    c, p = bs_price(S, K, T, r, s, q, "call"), bs_price(S, K, T, r, s, q, "put")
    assert np.isclose(c - p, S * np.exp(-q * T) - K * np.exp(-r * T), atol=1e-9)


@SETTINGS
@given(spot, strike, mat, rate, vol, div)
def test_no_arbitrage_bounds(S, K, T, r, s, q):
    c = float(bs_price(S, K, T, r, s, q, "call"))
    assert max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) - 1e-10 <= c <= S * np.exp(-q * T) + 1e-10


@SETTINGS
@given(spot, st.floats(40, 190), mat, rate, vol, div)
def test_monotone_and_convex_in_strike(S, K, T, r, s, q):
    h = 5.0
    c0, c1, c2 = (float(bs_price(S, k, T, r, s, q)) for k in (K, K + h, K + 2 * h))
    assert c1 <= c0 + 1e-10  # décroissant en K
    assert c0 - 2 * c1 + c2 >= -1e-9  # convexe en K (densité >= 0)
    assert (c0 - c1) / h <= np.exp(-r * T) + 1e-10  # pente bornée par l'actualisation


@SETTINGS
@given(spot, strike, mat, rate, vol, div)
def test_greek_signs(S, K, T, r, s, q):
    g = bs_greeks(S, K, T, r, s, q)
    assert 0 <= g["delta"] <= np.exp(-q * T) + 1e-12
    assert g["gamma"] >= 0 and g["vega"] >= 0 and g["dual_gamma"] >= 0
    # équation de Black-Scholes : Θ + ½σ²S²Γ + (r-q)SΔ - rV = 0
    pde = g["theta"] + 0.5 * s * s * S * S * g["gamma"] + (r - q) * S * g["delta"] - r * g["price"]
    assert abs(pde) < 1e-8 * max(1.0, float(g["price"]))


@SETTINGS
@given(spot, strike, mat, rate, st.floats(0.05, 0.8), div)
def test_implied_vol_roundtrip(S, K, T, r, s, q):
    F = S * np.exp((r - q) * T)
    otype = "call" if K >= F else "put"
    price = float(bs_price(S, K, T, r, s, q, otype))
    if price < 1e-8 * S:  # prix numériquement nul : IV non identifiable
        return
    assert abs(implied_vol(price, S, K, T, r, q, otype) - s) < 1e-7
