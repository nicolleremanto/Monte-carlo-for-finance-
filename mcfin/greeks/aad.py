"""Différentiation automatique adjointe (AAD) sur ruban, vectorisée numpy.

Principe (Giles & Glasserman 2006, « Smoking adjoints » ; Capriotti 2011)
--------------------------------------------------------------------------
Le pricer est une composition d'opérations élémentaires. Le mode *direct*
propage d/dθ pour UN paramètre θ à la fois : coût ∝ nombre de paramètres.
Le mode *adjoint* enregistre les opérations sur un ruban (« tape ») puis
propage en sens inverse les adjoints x̄ = ∂V/∂x :

    pour y = f(x1, x2) : x̄1 += ȳ · ∂f/∂x1,  x̄2 += ȳ · ∂f/∂x2

Toutes les sensibilités s'obtiennent en un seul balayage arrière, pour un
coût borné par ~4x le pricing (théorème de Baur-Strassen / « cheap gradient
principle ») — indépendamment du nombre de paramètres. C'est la technique
standard des banques pour les Greeks XVA (des milliers de sensibilités).

En Monte Carlo, l'AAD calcule l'estimateur *pathwise* : il exige un payoff
lipschitzien (sinon, lissage — cf. ``smooth_step``).

Implémentation : chaque nœud stocke ses parents et les fonctions
vecteur-jacobien (VJP). Les tableaux numpy (une valeur par trajectoire)
sont des « scalaires vectorisés » ; la diffusion (broadcasting) est
inversée par sommation dans la passe arrière.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = ["Tape", "Var", "exp", "log", "sqrt", "maximum", "where", "smooth_step",
           "mean", "aad_greeks"]


def _unbroadcast(g: np.ndarray, shape: tuple) -> np.ndarray:
    g = np.asarray(g)
    if g.shape == shape:
        return g
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for ax, n in enumerate(shape):
        if n == 1 and g.shape[ax] != 1:
            g = g.sum(axis=ax, keepdims=True)
    return g.reshape(shape)


class Tape:
    def __init__(self):
        self.shapes: list[tuple] = []
        self.parents: list[tuple] = []

    def _record(self, value, parents) -> "Var":
        self.shapes.append(np.shape(value))
        self.parents.append(tuple(parents))
        return Var(value, self, len(self.shapes) - 1)

    def variable(self, value) -> "Var":
        return self._record(np.asarray(value, dtype=float), ())

    def gradient(self, out: "Var", inputs: list["Var"]) -> list:
        adj: list = [None] * len(self.shapes)
        adj[out.idx] = np.ones(self.shapes[out.idx])
        for i in range(out.idx, -1, -1):
            g = adj[i]
            if g is None:
                continue
            for p, vjp in self.parents[i]:
                c = _unbroadcast(vjp(g), self.shapes[p])
                adj[p] = c if adj[p] is None else adj[p] + c
            if self.parents[i]:
                adj[i] = None  # libère la mémoire des adjoints intermédiaires
        return [adj[v.idx] if adj[v.idx] is not None else np.zeros(self.shapes[v.idx])
                for v in inputs]


class Var:
    """Variable enregistrée sur le ruban (valeur scalaire ou tableau)."""
    __slots__ = ("value", "tape", "idx")
    __array_priority__ = 1000
    __array_ufunc__ = None  # ndarray (op) Var -> délègue aux opérateurs réfléchis de Var

    def __init__(self, value, tape: Tape, idx: int):
        self.value, self.tape, self.idx = value, tape, idx

    # --- helpers -----------------------------------------------------------
    def _new(self, value, parents):
        return self.tape._record(value, parents)

    @staticmethod
    def _val(x):
        return x.value if isinstance(x, Var) else x

    def _binary(self, other, value, d_self, d_other):
        parents = [(self.idx, d_self)]
        if isinstance(other, Var):
            parents.append((other.idx, d_other))
        return self._new(value, parents)

    # --- opérateurs -------------------------------------------------------------
    def __add__(self, o):
        return self._binary(o, self.value + self._val(o), lambda g: g, lambda g: g)

    __radd__ = __add__

    def __sub__(self, o):
        return self._binary(o, self.value - self._val(o), lambda g: g, lambda g: -g)

    def __rsub__(self, o):
        return self._new(o - self.value, [(self.idx, lambda g: -g)])

    def __mul__(self, o):
        a, b = self.value, self._val(o)
        return self._binary(o, a * b, lambda g: g * b, lambda g: g * a)

    __rmul__ = __mul__

    def __truediv__(self, o):
        a, b = self.value, self._val(o)
        return self._binary(o, a / b, lambda g: g / b, lambda g: -g * a / (b * b))

    def __rtruediv__(self, o):
        a = self.value
        return self._new(o / a, [(self.idx, lambda g: -g * o / (a * a))])

    def __neg__(self):
        return self._new(-self.value, [(self.idx, lambda g: -g)])

    def __pow__(self, p: float):
        a = self.value
        return self._new(a**p, [(self.idx, lambda g: g * p * a ** (p - 1))])

    def __repr__(self):
        return f"Var({self.value!r})"


# --- fonctions (acceptent Var ou numpy) ------------------------------------------
def exp(x):
    if not isinstance(x, Var):
        return np.exp(x)
    e = np.exp(x.value)
    return x._new(e, [(x.idx, lambda g: g * e)])


def log(x):
    if not isinstance(x, Var):
        return np.log(x)
    a = x.value
    return x._new(np.log(a), [(x.idx, lambda g: g / a)])


def sqrt(x):
    """Racine avec dérivée nulle en 0 (évite 0·∞ en full truncation)."""
    if not isinstance(x, Var):
        return np.sqrt(x)
    s = np.sqrt(x.value)
    d = np.where(s > 0, 0.5 / np.where(s > 0, s, 1.0), 0.0)
    return x._new(s, [(x.idx, lambda g: g * d)])


def maximum(a, b):
    va, vb = Var._val(a), Var._val(b)
    if not isinstance(a, Var) and not isinstance(b, Var):
        return np.maximum(va, vb)
    mask = va >= vb
    tape = a.tape if isinstance(a, Var) else b.tape
    parents = []
    if isinstance(a, Var):
        parents.append((a.idx, lambda g: g * mask))
    if isinstance(b, Var):
        parents.append((b.idx, lambda g: g * ~mask))
    return tape._record(np.maximum(va, vb), parents)


def where(cond, a, b):
    va, vb = Var._val(a), Var._val(b)
    if not isinstance(a, Var) and not isinstance(b, Var):
        return np.where(cond, va, vb)
    tape = a.tape if isinstance(a, Var) else b.tape
    parents = []
    if isinstance(a, Var):
        parents.append((a.idx, lambda g: np.where(cond, g, 0.0)))
    if isinstance(b, Var):
        parents.append((b.idx, lambda g: np.where(cond, 0.0, g)))
    return tape._record(np.where(cond, va, vb), parents)


def mean(x, axis=None):
    if not isinstance(x, Var):
        return np.mean(x, axis=axis)
    shape = np.shape(x.value)
    n = np.prod(shape) if axis is None else shape[axis]

    def vjp(g):
        g = np.asarray(g)
        if axis is not None:
            g = np.expand_dims(g, axis)
        return np.broadcast_to(g / n, shape)
    return x._new(np.mean(x.value, axis=axis), [(x.idx, vjp)])


def smooth_step(x, eps: float):
    """Heaviside lissée 1/(1+e^{-x/eps}) : rend l'estimateur pathwise
    applicable aux digitales/barrières (biais O(eps²), variance O(1/eps))."""
    return 1.0 / (1.0 + exp(-x / eps))


def aad_greeks(pricer: Callable, params: dict[str, float], z: np.ndarray,
               batch_size: int = 20_000) -> dict:
    """Prix et gradient complet par AAD, par lots.

    ``pricer(p: dict[str, Var], z_batch) -> Var`` renvoie la moyenne des flux
    actualisés du lot. Le gradient de la moyenne globale est la moyenne
    pondérée des gradients par lot ; l'erreur standard est estimée par la
    dispersion inter-lots.
    """
    names = list(params)
    n = z.shape[0]
    prices, grads, weights = [], [], []
    for start in range(0, n, batch_size):
        zb = z[start:start + batch_size]
        tape = Tape()
        pv = {k: tape.variable(v) for k, v in params.items()}
        out = pricer(pv, zb)
        g = tape.gradient(out, [pv[k] for k in names])
        prices.append(float(out.value))
        grads.append([float(np.sum(x)) for x in g])
        weights.append(zb.shape[0])
    w = np.asarray(weights, float) / n
    prices, grads = np.asarray(prices), np.asarray(grads)
    res = {"price": float(w @ prices)}
    nb = len(weights)
    for i, k in enumerate(names):
        res[k] = float(w @ grads[:, i])
        if nb > 1:
            res[k + "_stderr"] = float(np.sqrt(np.sum(w**2 * (grads[:, i] - res[k]) ** 2)
                                               * nb / (nb - 1)))
    return res
