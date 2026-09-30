"""Couverture delta en temps discret : l'expérience qui justifie Black-Scholes.

On vend une option au prix Black-Scholes calculé avec la volatilité
implicite σ_i, et l'on se couvre en delta (volatilité de couverture σ_h) à
n dates, le sous-jacent suivant la dynamique *historique*
dS/S = μ dt + σ_r dW (mesure P). Le P&L final du vendeur couvert,

    Π_T = (prime - coûts) capitalisée + Σ Δ_k (S_{k+1} - S_k) (+ dividendes, financement) - Φ(S_T),

illustre les résultats théoriques suivants.

1. **Réplication** (σ_r = σ_i = σ_h, sans frais) : Π_T -> 0 quand n -> ∞,
   quel que soit μ (la tendance n'a pas de prix). En temps discret,
   l'erreur de couverture a un écart-type en O(n^{-1/2}) ; pour une option
   ATM, Kamal & Derman (1999) :  sd(Π_T) ≈ sqrt(π/4) · vega · σ / sqrt(n).

2. **Vol mal spécifiée** (couverture à σ_h = σ_i ≠ σ_r) : en temps continu,
   (formule « du P&L de gamma », El Karoui, Jeanblanc & Shreve 1998)

       Π_T = ½ ∫_0^T e^{r(T-t)} Γ_t^{σ_i} S_t² (σ_i² - σ_r²) dt

   Vendre la vol trop cher (σ_i > σ_r) rapporte, mais le gain dépend du
   chemin (via le gamma). Couvrir au contraire à σ_h = σ_r rend le P&L
   déterministe : (V(σ_i) - V(σ_r)) e^{rT}.

3. **Coûts de transaction proportionnels** (k = fourchette aller-retour,
   coût (k/2)|ΔΔ|S par transaction) : Leland (1985) montre qu'on couvre
   en moyenne les frais en utilisant la volatilité modifiée
       σ_L² = σ² (1 + sqrt(2/π) · k / (σ sqrt(Δt))).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..analytics.black_scholes import bs_greeks, bs_price

__all__ = ["HedgeResult", "simulate_delta_hedge", "leland_volatility",
           "kamal_derman_std"]


@dataclass
class HedgeResult:
    pnl: np.ndarray               # P&L final (en T) par trajectoire
    premium: float
    gamma_pnl: np.ndarray         # ½∫ e^{r(T-t)} Γ S² (σ_h² - σ_r²) dt discrétisé
    costs: np.ndarray             # frais de transaction cumulés (capitalisés)
    n_rebalancing: int

    @property
    def mean(self) -> float:
        return float(self.pnl.mean())

    @property
    def std(self) -> float:
        return float(self.pnl.std(ddof=1))


def leland_volatility(sigma: float, cost: float, dt: float) -> float:
    return float(sigma * np.sqrt(1.0 + np.sqrt(2.0 / np.pi) * cost / (sigma * np.sqrt(dt))))


def kamal_derman_std(S0, K, T, r, sigma, n_rebalancing, q=0.0) -> float:
    """Écart-type théorique de l'erreur de couverture discrète (ATM)."""
    vega = float(bs_greeks(S0, K, T, r, sigma, q)["vega"])
    return float(np.sqrt(np.pi / 4) * vega * sigma / np.sqrt(n_rebalancing) * np.exp(r * T))


def simulate_delta_hedge(S0: float, K: float, T: float, r: float, sigma_real: float,
                         sigma_implied: float, mu: float = 0.05, q: float = 0.0,
                         sigma_hedge: float | None = None, n_rebalancing: int = 52,
                         n_paths: int = 10_000, option_type: str = "call",
                         cost: float = 0.0, setup_costs: bool = True,
                         seed: int | None = 0) -> HedgeResult:
    """Vendeur d'une option, couvert en delta à n dates équiréparties.

    ``cost`` : fourchette relative aller-retour k (frais (k/2)|ΔΔ|S).
    ``setup_costs`` : facturer aussi l'achat initial du delta et le débouclage
    final (Leland ne couvre que les frais de *rebalancement*).
    Le P&L inclut les dividendes reçus sur les actions détenues et le
    financement du compte cash au taux r.
    """
    sigma_h = sigma_implied if sigma_hedge is None else sigma_hedge
    rng = np.random.default_rng(seed)
    dt = T / n_rebalancing
    z = rng.standard_normal((n_paths, n_rebalancing))
    growth = np.exp((mu - q - 0.5 * sigma_real**2) * dt + sigma_real * np.sqrt(dt) * z)
    S = S0 * np.concatenate([np.ones((n_paths, 1)), np.cumprod(growth, axis=1)], axis=1)

    premium = float(bs_price(S0, K, T, r, sigma_implied, q, option_type))
    cash = np.full(n_paths, premium)
    delta_prev = np.zeros(n_paths)
    costs = np.zeros(n_paths)
    gamma_pnl = np.zeros(n_paths)
    for k in range(n_rebalancing):
        tau = T - k * dt
        g = bs_greeks(S[:, k], K, tau, r, sigma_h, q, option_type)
        delta = g["delta"]
        trade = delta - delta_prev
        fee = 0.5 * cost * np.abs(trade) * S[:, k] if (k > 0 or setup_costs) else 0.0 * trade
        cash -= trade * S[:, k] + fee
        costs = costs * np.exp(r * dt) + fee
        gamma_pnl += 0.5 * np.exp(r * (T - k * dt)) * g["gamma"] * S[:, k] ** 2 \
            * (sigma_h**2 - sigma_real**2) * dt
        # capitalisation du cash et dividendes reçus sur la position
        cash = cash * np.exp(r * dt)
        cash += delta * S[:, k + 1] * (np.exp(q * dt) - 1.0)
        delta_prev = delta
    ST = S[:, -1]
    payoff = np.maximum((1 if option_type == "call" else -1) * (ST - K), 0.0)
    unwind_fee = 0.5 * cost * np.abs(delta_prev) * ST * float(setup_costs)
    pnl = cash + delta_prev * ST - unwind_fee - payoff
    return HedgeResult(pnl=pnl, premium=premium, gamma_pnl=gamma_pnl,
                       costs=costs + unwind_fee, n_rebalancing=n_rebalancing)
