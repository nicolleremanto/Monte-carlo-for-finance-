/* Worker WebAssembly : exécute la vraie librairie Python mcfin dans le navigateur.
 * Python 3.12 + numpy + scipy sont fournis par Pyodide ; les sources de mcfin
 * sont lues directement dans le dépôt (même origine que la page). Le calcul
 * tourne hors du fil principal : l'interface reste réactive. */
const PYODIDE_VERSION = "0.27.7";
importScripts(`https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/pyodide.js`);

let pyodide = null;
let bridgeCall = null;

async function fetchText(path) {
  const res = await fetch(path, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${path} : HTTP ${res.status}`);
  return res.text();
}

async function init() {
  const say = (msg, pct) => postMessage({ type: "progress", msg, pct });
  say("Démarrage de Python (WebAssembly)…", 5);
  pyodide = await loadPyodide();
  say("Chargement de numpy et scipy…", 30);
  const manifest = JSON.parse(await fetchText("manifest.json"));
  await pyodide.loadPackage(manifest.packages);
  say("Chargement de la librairie mcfin…", 80);
  const base = "/home/pyodide/";
  const sources = await Promise.all(manifest.files.map((f) => fetchText(`../${f}`)));
  manifest.files.forEach((f, i) => {
    const dir = base + f.split("/").slice(0, -1).join("/");
    pyodide.FS.mkdirTree(dir);
    pyodide.FS.writeFile(base + f, sources[i]);
  });
  pyodide.FS.writeFile(base + "bridge.py", await fetchText("bridge.py"));
  pyodide.runPython(`import sys; sys.path.insert(0, "${base}")`);
  bridgeCall = pyodide.runPython("import bridge; bridge.call");
  const version = pyodide.runPython("import sys, numpy, scipy; f'Python {sys.version.split()[0]} · numpy {numpy.__version__} · scipy {scipy.__version__}'");
  postMessage({ type: "ready", engine: `${version} (WebAssembly)` });
}

const ready = init().catch((err) => postMessage({ type: "fatal", error: String(err) }));

onmessage = async (event) => {
  const { id, name, args } = event.data;
  await ready;
  try {
    const out = bridgeCall(name, JSON.stringify(args));
    postMessage({ type: "result", id, payload: JSON.parse(out) });
  } catch (err) {
    postMessage({ type: "result", id, payload: { ok: false, error: String(err) } });
  }
};
