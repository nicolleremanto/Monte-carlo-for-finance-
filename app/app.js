/* mcfin — logique de l'interface.
 * Deux moteurs d'exécution, même code Python (app/bridge.py) :
 *  - serveur local (python app/serve.py) : détecté via /api/ping ;
 *  - Pyodide (WebAssembly) dans un Web Worker : page statique (GitHub Pages).
 * Forcer Pyodide : ?engine=pyodide */
"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const nf = (d) => new Intl.NumberFormat("fr-FR", { minimumFractionDigits: d, maximumFractionDigits: d });
const fmt = (x, d = 4) => (x === null || x === undefined || Number.isNaN(x) ? "—" : nf(d).format(x));
const fmtSigned = (x, d = 4) => (x === null || x === undefined ? "—" : (x >= 0 ? "+" : "") + nf(d).format(x));
const fmtSci = (x) => (x === null || x === undefined ? "—" : Math.abs(x) < 1e-3 && x !== 0 ? x.toExponential(1).replace(".", ",") : fmtSigned(x, 4));

/* ------------------------------------------------------------------ moteur */
const Runtime = {
  mode: null,
  worker: null,
  seq: 0,
  pending: new Map(),

  async init() {
    const forced = new URLSearchParams(location.search).get("engine");
    if (location.protocol === "file:") {
      setStatus("error", "Ouvrez via un serveur");
      showLoader("Fichier ouvert en local", "Le navigateur interdit l'exécution depuis file://. Lancez « python app/serve.py » à la racine du dépôt, ou utilisez la page GitHub Pages du projet.", 0);
      return;
    }
    if (forced !== "pyodide") {
      try {
        const res = await fetch("/api/ping", { signal: AbortSignal.timeout(1500), cache: "no-store" });
        if (res.ok) {
          const info = await res.json();
          this.mode = "server";
          engineLabel = `${info.engine} · serveur local`;
          setStatus("ready", engineLabel);
          enableButtons(true);
          return;
        }
      } catch (_) { /* pas de serveur : Pyodide */ }
    }
    this.mode = "pyodide";
    showLoader("Chargement du moteur de calcul", "Python, numpy et scipy s'exécutent directement dans votre navigateur (WebAssembly). Le premier chargement télécharge ~30 Mo puis reste en cache.", 3);
    setStatus("busy", "Chargement de Python…");
    this.worker = new Worker("worker.js");
    this.worker.onmessage = (ev) => this.onMessage(ev.data);
    this.worker.onerror = (ev) => fatal(`Erreur du moteur WebAssembly : ${ev.message || ev}`);
  },

  onMessage(msg) {
    if (msg.type === "progress") {
      showLoader(null, msg.msg, msg.pct);
      setStatus("busy", msg.msg);
    } else if (msg.type === "ready") {
      hideLoader();
      engineLabel = msg.engine;
      setStatus("ready", engineLabel);
      enableButtons(true);
    } else if (msg.type === "fatal") {
      fatal(`Impossible de démarrer Python dans le navigateur : ${msg.error}`);
    } else if (msg.type === "result") {
      const cb = this.pending.get(msg.id);
      this.pending.delete(msg.id);
      if (cb) cb(msg.payload);
    }
  },

  call(name, args) {
    if (this.mode === "server") {
      return fetch(`/api/${name}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(args) })
        .then((r) => r.json());
    }
    return new Promise((resolve) => {
      const id = ++this.seq;
      this.pending.set(id, resolve);
      this.worker.postMessage({ id, name, args });
    });
  },
};

let engineLabel = "";
function setStatus(state, text) {
  const el = $("#status");
  el.className = `status ${state}`;
  $("#status-text").textContent = text;
}
function showLoader(title, msg, pct) {
  $("#loader").hidden = false;
  if (title) $("#loader-title").textContent = title;
  if (msg) $("#loader-msg").textContent = msg;
  if (pct !== undefined) $("#loader-bar").style.width = `${Math.max(3, pct)}%`;
}
function hideLoader() { $("#loader").hidden = true; }
function fatal(message) {
  setStatus("error", "Moteur indisponible");
  showLoader("Moteur indisponible", message, 0);
}
function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { el.hidden = true; }, 7000);
}
function enableButtons(on) { $$(".primary, .secondary").forEach((b) => { b.disabled = !on; }); }

async function run(name, args, button) {
  enableButtons(false);
  const label = button.textContent;
  button.textContent = "Calcul…";
  setStatus("busy", "Calcul en cours…");
  let status = engineLabel;
  try {
    const out = await Runtime.call(name, args);
    if (!out.ok) { toast(out.error); return null; }
    status = `${engineLabel} · calcul ${fmt(out.wall, 2)} s`;
    return out.result;
  } catch (err) {
    toast(String(err));
    return null;
  } finally {
    button.textContent = label;
    enableButtons(true);
    setStatus("ready", status);
  }
}

/* ------------------------------------------------------------------ graphiques */
const charts = {};
function palette() { return [css("--series-1"), css("--series-2"), css("--series-3")]; }
function alpha(hex, a) {
  const n = parseInt(hex.replace("#", ""), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}
function axis(title, type = "linear", extra = {}) {
  return {
    type,
    title: { display: !!title, text: title, color: css("--ink-2") },
    grid: { color: css("--grid") },
    border: { color: css("--axis") },
    ticks: { color: css("--muted"), maxTicksLimit: 8 },
    ...extra,
  };
}
function draw(id, type, data, { x, y, legend = true, tooltip } = {}) {
  if (charts[id]) charts[id].destroy();
  Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
  Chart.defaults.color = css("--ink-2");
  charts[id] = new Chart(document.getElementById(id), {
    type,
    data,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      locale: "fr-FR",
      animation: { duration: 250 },
      interaction: type === "scatter" ? { mode: "nearest", intersect: true } : { mode: "index", intersect: false },
      plugins: {
        legend: { display: legend, labels: { color: css("--ink-2"), boxWidth: 14, boxHeight: 2, usePointStyle: false,
          filter: (item, d) => !d.datasets[item.datasetIndex].hideInLegend } },
        tooltip: { callbacks: tooltip || {}, filter: (item) => !item.dataset.hideInTooltip },
      },
      scales: { x, y },
      elements: { line: { borderWidth: 2, tension: 0 }, point: { radius: 0, hoverRadius: 4 } },
    },
  });
}
const line = (label, xs, ys, color, extra = {}) => ({
  label, data: xs.map((v, i) => ({ x: v, y: ys[i] })), borderColor: color, backgroundColor: color, ...extra,
});
function histogram(id, hist, color, xTitle) {
  const centers = hist.edges.slice(0, -1).map((e, i) => (e + hist.edges[i + 1]) / 2);
  draw(id, "bar", {
    labels: centers.map((c) => fmt(c, 2)),
    datasets: [{ label: "Trajectoires", data: hist.counts, backgroundColor: color, borderWidth: 0, barPercentage: 1, categoryPercentage: 0.94 }],
  }, { legend: false, x: axis(xTitle, "category", { ticks: { color: css("--muted"), maxTicksLimit: 7, maxRotation: 0 } }), y: axis("Nombre de trajectoires") });
}

/* ------------------------------------------------------------------ utilitaires UI */
function segValue(name) { return $(`.seg[data-name="${name}"] button.on`).dataset.value; }
$$(".seg").forEach((seg) => seg.addEventListener("click", (ev) => {
  const b = ev.target.closest("button");
  if (!b) return;
  seg.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
}));
const num = (id) => parseFloat($(id).value);
function kpis(target, tiles) {
  $(target).innerHTML = tiles.map(([label, value, sub, cls]) => `
    <div class="kpi"><div class="label">${label}</div><div class="value ${cls || ""} ${String(value).length > 9 ? "long" : ""}">${value}</div>${sub ? `<div class="sub-value">${sub}</div>` : ""}</div>`).join("");
}

/* navigation */
$$(".tab").forEach((t) => t.addEventListener("click", () => {
  $$(".tab").forEach((x) => { x.classList.toggle("active", x === t); x.setAttribute("aria-selected", x === t); });
  $$(".page").forEach((p) => p.classList.toggle("active", p.id === `page-${t.dataset.page}`));
  try { localStorage.setItem("mcfin-tab", t.dataset.page); } catch (_) { /* stockage indisponible */ }
}));
let bsSub = "overview";
const BS_HINTS = {
  overview: "Formule fermée, Monte Carlo, QMC, EDP de Crank-Nicolson et arbre binomial sur la même option.",
  american: "Américaine : EDP (Brennan-Schwartz), arbre de Leisen-Reimer et Longstaff-Schwartz (50 dates).",
  hedge: "On vend l'option à la vol σ et on se couvre en delta ; le sous-jacent suit une vol réalisée σᵣ et une tendance μ.",
};
$$(".subtab").forEach((t) => t.addEventListener("click", () => {
  bsSub = t.dataset.sub;
  $$(".subtab").forEach((x) => x.classList.toggle("active", x === t));
  $$("#page-bs .sub").forEach((s) => s.classList.toggle("active", s.id === `bs-${bsSub}`));
  $("#bs-hedge-fields").hidden = bsSub !== "hedge";
  $("#bs-hint").textContent = BS_HINTS[bsSub];
}));

/* ------------------------------------------------------------------ Black-Scholes */
const GREEK_INFO = {
  delta: ["Δ", "∂V/∂S — couverture en sous-jacent"], gamma: ["Γ", "∂²V/∂S² — convexité"],
  vega: ["ν", "∂V/∂σ — risque de volatilité"], theta: ["Θ", "∂V/∂t — portage (par an)"],
  rho: ["ρ", "∂V/∂r — taux"], epsilon: ["ε", "∂V/∂q — dividende"],
  vanna: ["vanna", "∂²V/∂S∂σ"], volga: ["volga", "∂²V/∂σ²"], charm: ["charm", "∂Δ/∂t"],
  speed: ["speed", "∂Γ/∂S"], zomma: ["zomma", "∂Γ/∂σ"], color: ["color", "∂Γ/∂t"],
  ultima: ["ultima", "∂³V/∂σ³"], dual_delta: ["dual Δ", "∂V/∂K"], dual_gamma: ["dual Γ", "∂²V/∂K² (densité RN)"],
};
function bsArgs() {
  return {
    S0: num("#bs-S0"), K: num("#bs-K"), T: num("#bs-T"), sigma: num("#bs-sigma"), r: num("#bs-r"), q: num("#bs-q"),
    option_type: segValue("bs-type"), n_paths: num("#bs-n"),
    sigma_real: num("#bs-sreal"), mu: num("#bs-mu"), n_rebalancing: num("#bs-nreb"), cost: num("#bs-cost"),
  };
}

function renderOverview(r, args) {
  const g = r.greeks;
  kpis("#bs-kpis", [
    ["Prix (formule fermée)", fmt(r.price, 4), `${args.option_type === "call" ? "Call" : "Put"} K = ${fmt(args.K, 2)}`],
    ["Delta Δ", fmt(g.delta, 4)], ["Gamma Γ", fmt(g.gamma, 5)],
    ["Vega ν (1 pt de vol)", fmt(g.vega / 100, 4)], ["Theta Θ (par jour)", fmt(g.theta / 365, 4)],
  ]);
  $("#bs-compare").innerHTML = `<tr><th>Méthode</th><th class="num">Prix</th><th class="num">Écart</th><th class="num">Err. std</th><th class="num">Temps</th></tr>` +
    r.compare.map((row) => `<tr><td>${row["méthode"]}</td><td class="num">${fmt(row.prix, 5)}</td>
      <td class="num">${row["méthode"] === "Formule fermée" ? "référence" : fmtSci(row["écart"])}</td>
      <td class="num">${row.err_std === null ? "—" : fmt(row.err_std, 5)}</td><td class="num">${fmt(row.temps_s * 1000, 1)} ms</td></tr>`).join("");
  $("#bs-greeks").innerHTML = `<tr><th>Greek</th><th class="num">Valeur</th><th>Définition</th></tr>` +
    Object.entries(GREEK_INFO).map(([k, [sym, desc]]) => `<tr><td>${sym}</td><td class="num">${fmt(g[k], 5)}</td><td class="desc">${desc}</td></tr>`).join("");
  const [c1, c2] = palette();
  const c = r.curve;
  draw("c-bs-price", "line", { datasets: [
    line("Prix Black-Scholes", c.S, c.price, c1),
    line("Valeur intrinsèque", c.S, c.intrinsic, c2, { borderDash: [5, 4] }),
  ] }, { x: axis("Spot S", "linear"), y: axis("Valeur") });
  draw("c-bs-delta", "line", { datasets: [line("Delta", c.S, c.delta, c1)] }, { legend: false, x: axis("Spot S"), y: axis("Δ") });
  draw("c-bs-gamma", "line", { datasets: [line("Gamma", c.S, c.gamma, c1)] }, { legend: false, x: axis("Spot S"), y: axis("Γ") });
}

function renderAmerican(r, args) {
  kpis("#am-kpis", [
    ["Européenne (formule)", fmt(r.european, 4)],
    ["Américaine — EDP", fmt(r.american_pde, 4), "Crank-Nicolson + Brennan-Schwartz"],
    ["Américaine — arbre", fmt(r.american_tree, 4), "Leisen-Reimer, 1001 pas"],
    ["Longstaff-Schwartz", fmt(r.lsm, 4), `± ${fmt(r.lsm_stderr, 4)} (bermudéenne, 50 dates)`],
    ["Prime d'exercice anticipé", fmt(r.premium, 4), r.premium < 1e-3 ? "exercice anticipé sans valeur" : ""],
  ]);
  const [c1, c2] = palette();
  if (r.boundary) {
    const b = r.boundary;
    draw("c-am-boundary", "line", { datasets: [
      line("Frontière S*(t)", b.t, b.S, c1),
      line("Strike K", [0, args.T], [args.K, args.K], c2, { borderDash: [5, 4], borderWidth: 1.5 }),
    ] }, { x: axis("Temps t (années)", "linear", { min: 0, max: args.T }), y: axis("Spot") });
  } else {
    draw("c-am-boundary", "line", { datasets: [] }, { legend: false, x: axis("Temps t (années)"), y: axis("Spot") });
    toast("Pas de région d'exercice anticipé pour cette option (ex. call sans dividende).");
  }
}

function renderHedge(r, args) {
  const tiles = [
    ["Prime encaissée", fmt(r.premium, 4), `vol implicite ${fmt(args.sigma * 100, 1)} %`],
    ["P&L moyen en T", fmtSigned(r.mean, 4), `± ${fmt(r.mean_stderr, 4)} (err. std)`],
    ["Écart-type du P&L", fmt(r.std, 4), r.kamal_derman ? `Kamal-Derman : ${fmt(r.kamal_derman, 4)}` : `${args.n_rebalancing} rebalancements`],
    ["P&L de gamma (théorie)", fmtSigned(r.gamma_pnl_mean, 4), "½ ∫ e^(r(T−t)) Γ S² (σ² − σᵣ²) dt"],
    ["Frais moyens", fmt(r.costs_mean, 4), `k = ${fmt(args.cost * 100, 2)} %`],
  ];
  if (r.leland) tiles.push(["Vol de Leland σ_L", `${fmt(r.leland.sigma_leland * 100, 2)} %`,
    `P&L moyen ${fmtSigned(r.leland.mean_bs, 3)} (BS) → ${fmtSigned(r.leland.mean_leland, 3)}`]);
  kpis("#hd-kpis", tiles);
  const [c1, c2] = palette();
  histogram("c-hd-hist", r.hist, c1, "P&L du vendeur couvert");
  const s = r.scaling;
  const ds = [line("Simulation", s.n, s.std, c1, { pointRadius: 3 })];
  if (s.theory) ds.push(line("Kamal-Derman √(π/4)·vega·σ/√n", s.n, s.theory, c2, { borderDash: [5, 4] }));
  draw("c-hd-scaling", "line", { datasets: ds }, { x: axis("Nombre de rebalancements", "logarithmic"), y: axis("Écart-type du P&L", "logarithmic") });
  const card = $("#hd-scatter-card");
  card.hidden = !r.scatter;
  if (r.scatter) {
    const xs = r.scatter.x;
    const lo = Math.min(...xs), hi = Math.max(...xs);
    draw("c-hd-scatter", "scatter", { datasets: [
      { label: "Trajectoires", data: xs.map((v, i) => ({ x: v, y: r.scatter.y[i] })), backgroundColor: alpha(c1, 0.45), pointRadius: 2.5 },
      { type: "line", label: "Réalisé = théorie", data: [{ x: lo, y: lo }, { x: hi, y: hi }], borderColor: c2, borderDash: [5, 4], borderWidth: 1.5, pointRadius: 0 },
    ] }, { x: axis("P&L de gamma (théorie)"), y: axis("P&L réalisé") });
  }
  const same = Math.abs(args.sigma_real - args.sigma) < 1e-12;
  $("#hd-explain").textContent = same
    ? "Vol réalisée = vol implicite : le P&L est d'espérance nulle quelle que soit la tendance μ, et son écart-type décroît en 1/√n (réplication imparfaite en temps discret)."
    : `Vol vendue ${fmt(args.sigma * 100, 1)} %, réalisée ${fmt(args.sigma_real * 100, 1)} % : le gain moyen suit le P&L de gamma, mais il dépend du chemin (poids ΓS²). Couvrir à la vol réalisée le rendrait déterministe.`;
}

$("#bs-run").addEventListener("click", async (ev) => {
  const args = bsArgs();
  const fn = { overview: "bs_overview", american: "bs_american", hedge: "bs_hedging" }[bsSub];
  const payload = bsSub === "hedge" ? { ...args, n_paths: num("#bs-hn") } : args;
  const r = await run(fn, payload, ev.currentTarget);
  if (!r) return;
  ({ overview: renderOverview, american: renderAmerican, hedge: renderHedge })[bsSub](r, args);
});

/* ------------------------------------------------------------------ moteur Monte Carlo */
const MODEL_FIELDS = {
  bs: [["spot", "Spot S₀", 100], ["vol", "Volatilité σ", 0.2], ["rate", "Taux r", 0.03], ["div", "Dividende q", 0]],
  heston: [["spot", "Spot S₀", 100], ["v0", "Variance v₀", 0.04], ["kappa", "Rappel κ", 1.5], ["theta", "Variance θ (long t.)", 0.04],
    ["xi", "Vol de vol ξ", 0.6], ["rho", "Corrélation ρ", -0.7], ["rate", "Taux r", 0.03], ["div", "Dividende q", 0]],
  merton: [["spot", "Spot S₀", 100], ["vol", "Volatilité σ", 0.2], ["lam", "Intensité λ", 0.5], ["mu_j", "Saut moyen μ_J", -0.1],
    ["sigma_j", "Vol des sauts δ", 0.15], ["rate", "Taux r", 0.03], ["div", "Dividende q", 0]],
  rbergomi: [["spot", "Spot S₀", 100], ["xi0", "Variance forward ξ₀", 0.04], ["eta", "Vol de vol η", 1.9], ["hurst", "Hurst H", 0.1],
    ["rho", "Corrélation ρ", -0.9], ["rate", "Taux r", 0.03], ["div", "Dividende q", 0]],
  localvol: [["spot", "Spot S₀", 100], ["ssvi_sigma0", "Vol ATM courte", 0.18], ["ssvi_sigma_inf", "Vol ATM longue", 0.22],
    ["ssvi_rho", "Skew ρ (SSVI)", -0.6], ["ssvi_eta", "Courbure η (SSVI)", 1.0], ["rate", "Taux r", 0.03], ["div", "Dividende q", 0]],
};
const PRODUCT_FIELDS = {
  european: [["strike", "Strike K", 100], ["maturity", "Maturité T", 1]],
  digital: [["strike", "Strike K", 100], ["maturity", "Maturité T", 1]],
  asian: [["strike", "Strike K", 100], ["maturity", "Maturité T", 1], ["n_fixings", "Fixings", 12]],
  barrier: [["strike", "Strike K", 100], ["barrier", "Barrière H", 80], ["maturity", "Maturité T", 1], ["n_monitoring", "Dates", 50]],
  lookback: [["maturity", "Maturité T", 1], ["n_monitoring", "Dates", 100]],
};
const MODEL_TEXT = {
  bs: "Black-Scholes : simulation exacte du log-prix (aucun biais de discrétisation).",
  heston: "Heston : variance CIR simulée par le schéma Quadratic-Exponential d'Andersen avec correction de martingale (pas de 1/32 an).",
  merton: "Merton : diffusion + sauts lognormaux poissonniens, simulation exacte aux dates d'observation.",
  rbergomi: "Rough Bergomi : volatilité « rugueuse » (processus de Volterra, H ≈ 0,1) par le schéma hybride + FFT ; pas de formule fermée.",
  localvol: "Volatilité locale de Dupire calculée analytiquement sur une nappe SSVI sans arbitrage ; reprice les vanilles de la nappe.",
};
function fieldHTML([key, label, def], values) {
  const v = values[key] ?? def;
  return `<label>${label}<input type="number" data-key="${key}" value="${v}" step="any"></label>`;
}
function currentValues(container) {
  const out = {};
  $$(`${container} input[data-key], ${container} select[data-key]`).forEach((el) => { out[el.dataset.key] = el.value; });
  return out;
}
function renderModelFields() {
  const vals = currentValues("#mc-model-fields");
  $("#mc-model-fields").innerHTML = MODEL_FIELDS[$("#mc-model").value].map((f) => fieldHTML(f, vals)).join("");
}
function renderProductFields() {
  const vals = currentValues("#mc-product-fields");
  const kind = $("#mc-product").value;
  let html = PRODUCT_FIELDS[kind].map((f) => fieldHTML(f, vals)).join("");
  if (kind === "barrier") {
    const bt = vals.barrier_type || "down-and-out";
    html += `<label style="grid-column: 1 / -1">Type de barrière<select data-key="barrier_type">
      ${["down-and-out", "down-and-in", "up-and-out", "up-and-in"].map((t) => `<option ${t === bt ? "selected" : ""}>${t}</option>`).join("")}
    </select></label>`;
  }
  $("#mc-product-fields").innerHTML = html;
}
$("#mc-model").addEventListener("change", renderModelFields);
$("#mc-product").addEventListener("change", renderProductFields);
renderModelFields();
renderProductFields();

function mcArgs() {
  const args = { model: $("#mc-model").value, product: $("#mc-product").value, option_type: segValue("mc-type"),
    method: segValue("mc-method"), n_paths: num("#mc-n"), seed: num("#mc-seed"),
    antithetic: $("#mc-anti").checked, control_variate: $("#mc-cv").checked };
  $$("#mc-model-fields [data-key], #mc-product-fields [data-key]").forEach((el) => {
    args[el.dataset.key] = el.tagName === "SELECT" ? el.value : parseFloat(el.value);
  });
  return args;
}

$("#mc-run").addEventListener("click", async (ev) => {
  const args = mcArgs();
  const r = await run("mc_price", args, ev.currentTarget);
  if (!r) return;
  $("#mc-conv-card").hidden = true; // l'étude de convergence précédente ne correspond plus
  const [lo, hi] = r.ci;
  const tiles = [
    ["Prix Monte Carlo", fmt(r.price, 4), `${nf(0).format(r.n_paths)} trajectoires · ${r.method}`],
    ["Erreur standard", fmt(r.stderr, 4), `${fmt(100 * r.stderr / Math.abs(r.price || 1), 2)} % du prix`],
    ["IC à 95 %", `[${fmt(lo, 3)} ; ${fmt(hi, 3)}]`, r.method.startsWith("sobol") ? "Student, 15 ddl (RQMC)" : "gaussien (TCL)"],
  ];
  if (r.reference !== null) {
    const ok = Math.abs(r.z) < 3;
    tiles.push(["Référence", fmt(r.reference, 4), r.reference_label]);
    tiles.push(["Écart en erreurs std", fmtSigned(r.z, 2), ok ? "compatible (|z| < 3)" : "écart significatif", ok ? "ok" : "ko"]);
  }
  if (r.variance_reduction) tiles.push(["Réduction de variance", `× ${fmt(r.variance_reduction, 1)}`, r.cv_label]);
  const dur = r.elapsed < 0.1 ? `${fmt(r.elapsed * 1000, 1)} ms` : `${fmt(r.elapsed, 2)} s`;
  tiles.push(["Temps de calcul", dur, Runtime.mode === "server" ? "CPython" : "WebAssembly"]);
  kpis("#mc-kpis", tiles);

  const [c1, c2] = palette();
  const t = r.paths.t;
  const ds = r.paths.S.map((path, i) => line(`trajectoire ${i + 1}`, t, path, alpha(c1, 0.5),
    { borderWidth: 1.2, hideInLegend: true, hideInTooltip: true }));
  if (r.barrier !== null) ds.push(line("Barrière H", [0, t[t.length - 1]], [r.barrier, r.barrier], c2, { borderDash: [6, 4], borderWidth: 1.5 }));
  draw("c-mc-paths", "line", { datasets: ds }, { legend: r.barrier !== null, x: axis("Temps (années)"), y: axis("Spot") });
  histogram("c-mc-hist", r.hist, c1, "Flux actualisé par trajectoire");
  $("#mc-explain").textContent = MODEL_TEXT[args.model] + (r.reference === null ? " Aucune formule fermée pour ce couple modèle/produit : l'intervalle de confiance est la seule mesure de précision." : "");
});

$("#mc-conv").addEventListener("click", async (ev) => {
  const args = mcArgs();
  const r = await run("mc_convergence", args, ev.currentTarget);
  if (!r) return;
  $("#mc-conv-card").hidden = false;
  const cols = palette();
  draw("c-mc-conv", "line", { datasets: r.series.map((s, i) =>
    line(`${s.label} (pente ${fmt(s.slope, 2)})`, r.n, s.stderr, cols[i], { pointRadius: 3 })) },
  { x: axis("Nombre de trajectoires N", "logarithmic"), y: axis("Erreur standard", "logarithmic") });
  $("#mc-conv-note").textContent = "échelles log — pente −0,5 en Monte Carlo, proche de −1 en QMC";
  $("#mc-conv-card").scrollIntoView({ behavior: "smooth", block: "nearest" });
});

/* ------------------------------------------------------------------ démarrage */
try {
  const saved = localStorage.getItem("mcfin-tab");
  if (saved) $(`.tab[data-page="${saved}"]`)?.click();
} catch (_) { /* stockage indisponible */ }
enableButtons(false);
Runtime.init();
