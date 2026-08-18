const boardEl = document.getElementById("board");
const statusEl = document.getElementById("status");
const infoEl = document.getElementById("info");
const modeEl = document.getElementById("mode");
const ckptEl = document.getElementById("checkpoint");
const simsEl = document.getElementById("simulations");
const stageEl = document.getElementById("board-stage");

let state = null;
let selected = null;
let busy = false;
let autoplay = false;

function rolesFromMode(mode) {
  if (mode === "hh") return { red: "human", black: "human" };
  if (mode === "hb") return { red: "ai", black: "human" };
  if (mode === "aa") return { red: "ai", black: "ai" };
  return { red: "human", black: "ai" };
}

function lastFromTo(last) {
  if (!last) return null;
  if (typeof last === "string" && last.length >= 4) {
    return { from: last.slice(0, 2), to: last.slice(2, 4) };
  }
  return last;
}

function inPalace(file, rank) {
  return file >= 3 && file <= 5 && (rank <= 2 || rank >= 7);
}

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

function loserKingIccs(state) {
  if (!state.over) return null;
  const loser = state.outcome === "红胜" ? "black" : state.outcome === "黑胜" ? "red" : null;
  if (!loser) return null;
  for (const row of state.squares) {
    for (const cell of row) {
      if (cell.color === loser && (cell.glyph === "帅" || cell.glyph === "将")) return cell.iccs;
    }
  }
  return null;
}

async function api(path, body) {
  const opt = body === undefined
    ? { method: "GET" }
    : {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      };
  const res = await fetch(path, opt);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function currentRole() {
  if (!state) return "human";
  return state.side === "red" ? state.red : state.black;
}

function render() {
  if (!state) return;
  boardEl.innerHTML = "";
  const last = lastFromTo(state.last_move);
  const legal = selected && state.legal_from[selected] ? state.legal_from[selected] : [];
  const loser = loserKingIccs(state);
  for (const row of state.squares) {
    for (const cell of row) {
      const sq = document.createElement("div");
      sq.className = `sq file${cell.file} rank${cell.rank}`;
      if (cell.rank === 5) sq.classList.add("river-n");
      if (cell.rank === 4) sq.classList.add("river-s");
      if (inPalace(cell.file, cell.rank)) sq.classList.add("palace");
      if (!state.over && last && (cell.iccs === last.from || cell.iccs === last.to)) sq.classList.add("last");
      if (selected === cell.iccs) sq.classList.add("selected");
      if (loser && cell.iccs === loser) sq.classList.add("loser");
      if (legal.includes(cell.iccs)) {
        sq.classList.add("legal");
        const mark = document.createElement("div");
        mark.className = "mark";
        sq.appendChild(mark);
      }
      sq.dataset.iccs = cell.iccs;
      if (cell.glyph) {
        const p = document.createElement("div");
        p.className = `piece ${cell.color}`;
        p.textContent = cell.glyph;
        sq.appendChild(p);
      }
      sq.addEventListener("click", () => onSquare(cell.iccs));
      boardEl.appendChild(sq);
    }
  }
  applyGameOverUi(state);
  const err = state.error ? ` · ${state.error}` : "";
  if (state.over) {
    statusEl.insertAdjacentText("beforeend", err);
  } else {
    const check = state.in_check ? "（将军）" : "";
    statusEl.textContent = `轮到 ${state.side === "red" ? "红" : "黑"}` + check + err;
  }
  infoEl.textContent = [
    `FEN: ${state.fen}`,
    `步数: ${state.ply}  半回合: ${state.halfmove}`,
    `红: ${state.red}  黑: ${state.black}`,
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
  if (!selected) {
    if (state.legal_from[iccs]) selected = iccs;
    render();
    return;
  }
  if (selected === iccs) {
    selected = null;
    render();
    return;
  }
  const dests = state.legal_from[selected] || [];
  if (!dests.includes(iccs)) {
    selected = state.legal_from[iccs] ? iccs : null;
    render();
    return;
  }
  await doMove(selected + iccs);
}

async function doMove(iccs) {
  busy = true;
  try {
    state = await api("/api/move", { iccs });
    selected = null;
    render();
    await maybeAi();
  } finally {
    busy = false;
  }
}

async function maybeAi() {
  if (!state || state.over) {
    autoplay = false;
    return;
  }
  const bothAi = state.red === "ai" && state.black === "ai";
  if (bothAi && !autoplay) return;
  if (currentRole() !== "ai") return;
  busy = true;
  try {
    state = await api("/api/ai", {});
    selected = null;
    render();
  } finally {
    busy = false;
  }
  if (autoplay && state && !state.over && currentRole() === "ai") {
    await maybeAi();
  }
}

async function newGame() {
  autoplay = false;
  const roles = rolesFromMode(modeEl.value);
  busy = true;
  try {
    state = await api("/api/new", {
      red: roles.red,
      black: roles.black,
      simulations: Number(simsEl.value) || 80,
      checkpoint: ckptEl.value.trim(),
    });
    selected = null;
    render();
    syncForm();
  } finally {
    busy = false;
  }
  if (currentRole() === "ai") await maybeAi();
}

async function undo(body) {
  autoplay = false;
  busy = true;
  try {
    state = await api("/api/undo", body);
    selected = null;
    render();
  } finally {
    busy = false;
  }
}

document.getElementById("btn-new").onclick = newGame;
document.getElementById("btn-undo").onclick = () => undo({ plies: 1 });
document.getElementById("btn-undo-turn").onclick = () => undo({ human_turn: true, plies: 2 });
document.getElementById("btn-ai").onclick = async () => {
  if (busy || !state || state.over) return;
  autoplay = false;
  busy = true;
  try {
    state = await api("/api/ai", {});
    selected = null;
    render();
  } finally {
    busy = false;
  }
};
document.getElementById("btn-auto").onclick = async () => {
  if (!state) return;
  autoplay = !autoplay;
  render();
  if (autoplay) await maybeAi();
};

(async function init() {
  try {
    state = await api("/api/state");
    syncForm();
    render();
  } catch (e) {
    statusEl.textContent = "无法连接服务器: " + e.message;
  }
})();
