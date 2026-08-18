const boardEl = document.getElementById("board");
const statusEl = document.getElementById("status");
const infoEl = document.getElementById("info");
const modeEl = document.getElementById("mode");
const ckptEl = document.getElementById("checkpoint");
const simsEl = document.getElementById("simulations");
const stageEl = document.getElementById("board-stage");
const ranksEl = document.getElementById("ranks");
const filesEl = document.getElementById("files");

let state = null;
let busy = false;
let autoplay = false;

function resultClass(outcome) {
  if (outcome === "红胜") return "win-red";
  if (outcome === "白胜") return "win-white";
  if (outcome === "黑胜") return "win-black";
  if (outcome === "和棋") return "draw";
  return "draw";
}

function applyGameOverUi(state) {
  const kinds = ["over", "win-red", "win-white", "win-black", "draw"];
  statusEl.classList.remove(...kinds);
  stageEl.classList.toggle("over", !!state.over);
  document.getElementById("btn-ai").disabled = !state || state.over;
  document.getElementById("btn-auto").disabled = !state || state.over;
  if (!state.over) return;
  statusEl.classList.add("over", resultClass(state.outcome));
  statusEl.textContent = "";
  statusEl.append(state.outcome || "终局");
  if (state.reason) {
    const reason = document.createElement("span");
    reason.className = "reason";
    reason.textContent = state.reason;
    statusEl.append(reason);
  }
}

function rolesFromMode(mode) {
  if (mode === "hh") return { red: "human", black: "human" };
  if (mode === "hr") return { red: "human", black: "ai" };
  if (mode === "aa") return { red: "ai", black: "ai" };
  return { red: "ai", black: "human" };
}

async function api(path, body) {
  const opt = body === undefined
    ? { method: "GET" }
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch(path, opt);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function currentRole() {
  if (!state) return "human";
  return state.side === "black" ? state.black : state.red;
}

function renderAxes(size) {
  ranksEl.replaceChildren();
  filesEl.replaceChildren();
  for (let rank = size - 1; rank >= 0; rank--) {
    const s = document.createElement("span");
    s.textContent = String(rank);
    ranksEl.append(s);
  }
  const spacer = document.createElement("span");
  filesEl.append(spacer);
  for (let file = 0; file < size; file++) {
    const s = document.createElement("span");
    s.textContent = String.fromCharCode("a".charCodeAt(0) + file);
    filesEl.append(s);
  }
}

function render() {
  if (!state) return;
  const size = state.size;
  document.documentElement.style.setProperty("--n", String(size));
  renderAxes(size);
  boardEl.innerHTML = "";
  const last = state.last_move ? state.last_move.iccs : "";
  const legal = new Set(state.legal || []);
  const winning = new Set(state.winning || []);
  for (const row of state.squares) {
    for (const cell of row) {
      const el = document.createElement("div");
      el.className = "cell";
      el.title = cell.iccs;
      el.dataset.iccs = cell.iccs;
      if (legal.has(cell.iccs)) el.classList.add("legal");
      if (cell.iccs === last) el.classList.add("last");
      if (winning.has(cell.iccs)) el.classList.add("win");
      if (cell.color) {
        const s = document.createElement("div");
        s.className = `stone ${cell.color}`;
        el.appendChild(s);
      }
      el.addEventListener("click", () => onSquare(cell.iccs));
      boardEl.appendChild(el);
    }
  }
  applyGameOverUi(state);
  const err = state.error ? ` · ${state.error}` : "";
  if (state.over) {
    statusEl.insertAdjacentText("beforeend", err);
  } else {
    statusEl.textContent = `轮到 ${state.side === "black" ? "黑" : "白"}` + err;
  }
  infoEl.textContent = [
    `FEN: ${state.fen}`,
    `步数: ${state.ply}`,
    `黑: ${state.black}  白: ${state.red}`,
    `模拟: ${state.simulations}`,
    state.checkpoint ? `权重: ${state.checkpoint}` : "权重: （均匀先验）",
    state.history.length ? `着法: ${state.history.join(" ")}` : "着法: （开局）",
  ].join("\n");
  document.getElementById("btn-auto").textContent = autoplay ? "暂停机机" : "机机自动";
}

function syncForm() {
  if (!state) return;
  ckptEl.value = state.checkpoint || "";
  if (state.simulations) simsEl.value = state.simulations;
}

async function onSquare(iccs) {
  if (busy || !state || state.over) return;
  if (currentRole() !== "human") return;
  if (!(state.legal || []).includes(iccs)) return;
  await doMove(iccs);
}

async function doMove(iccs) {
  busy = true;
  try {
    state = await api("/api/move", { iccs });
    render();
    await maybeAi();
  } finally {
    busy = false;
  }
}

async function maybeAi() {
  if (!state || state.over) return;
  if (currentRole() !== "ai") return;
  state = await api("/api/ai", {});
  render();
  if (autoplay && currentRole() === "ai" && !state.over) {
    await maybeAi();
  }
}

async function refresh() {
  state = await api("/api/state");
  syncForm();
  render();
  await maybeAi();
}

document.getElementById("btn-new").addEventListener("click", async () => {
  const roles = rolesFromMode(modeEl.value);
  state = await api("/api/new", {
    red: roles.red,
    black: roles.black,
    simulations: Number(simsEl.value),
    checkpoint: (ckptEl.value || "").trim(),
  });
  autoplay = false;
  render();
  syncForm();
  await maybeAi();
});
document.getElementById("btn-undo").addEventListener("click", async () => {
  state = await api("/api/undo", { plies: 1 });
  render();
});
document.getElementById("btn-undo-turn").addEventListener("click", async () => {
  state = await api("/api/undo", { human_turn: true });
  render();
});
document.getElementById("btn-ai").addEventListener("click", async () => {
  state = await api("/api/ai", {});
  render();
});
document.getElementById("btn-auto").addEventListener("click", async () => {
  autoplay = !autoplay;
  render();
  if (autoplay) await maybeAi();
});

refresh().catch((err) => {
  statusEl.textContent = `加载失败：${err}`;
});
