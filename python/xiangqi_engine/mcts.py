"""AlphaZero MCTS (PUCT). Hyperparameters come from config/default.json.

Each search owns its Board (make/unmake). Do not share one MCTS or one Board
across threads. A network may be shared if NetworkEvaluator is given a lock.

When mcts.batch_size > 1, one search collects that many leaves, applies
virtual loss so PUCT does not collapse onto one edge, and evaluates them in
a single network forward. batch_size == 1 keeps the original sequential path
(no virtual loss) so tests that expect visit concentration still hold.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from xiangqi_engine._xiangqi import BLACK, RED, Outcome
from xiangqi_engine.config import Cfg, load_config
from xiangqi_engine.encode import Encoder


def terminal_value(board) -> float | None:
    """Value for the player to move, or None if the game is still going."""
    term = board.terminal()
    if term.outcome == Outcome.ONGOING:
        return None
    if term.outcome == Outcome.DRAW:
        return 0.0
    if term.outcome == Outcome.RED_WIN:
        return 1.0 if board.side_to_move() == RED else -1.0
    return 1.0 if board.side_to_move() == BLACK else -1.0


class Evaluator(Protocol):
    def evaluate(
        self, board, history: list
    ) -> tuple[list[int], np.ndarray, float]:
        """Return (legal action indices, priors over those actions, value)."""

    def evaluate_many(
        self, encodings: list, legal_lists: list[list[int]]
    ) -> list[tuple[np.ndarray, float]]:
        """Priors (over each legal list) and values for a batch of encoded leaves."""


class UniformEvaluator:
    """Equal priors, v=0. Used to test search structure without a network."""

    def __init__(self, encoder: Encoder):
        self.encoder = encoder

    def evaluate(self, board, history: list) -> tuple[list[int], np.ndarray, float]:
        del history
        legal = [int(i) for i in self.encoder.legal_action_indices(board)]
        prior, value = self.evaluate_many([None], [legal])[0]
        return legal, prior, value

    def evaluate_many(
        self, encodings: list, legal_lists: list[list[int]]
    ) -> list[tuple[np.ndarray, float]]:
        del encodings
        out: list[tuple[np.ndarray, float]] = []
        for legal in legal_lists:
            if not legal:
                out.append((np.zeros(0, dtype=np.float64), 0.0))
            else:
                out.append((np.full(len(legal), 1.0 / len(legal), dtype=np.float64), 0.0))
        return out


class NetworkEvaluator:
    def __init__(self, net, encoder: Encoder, device: str | None = None, lock=None):
        import torch

        self.net = net
        self.encoder = encoder
        self.device = device or next(net.parameters()).device
        self.lock = lock
        self._torch = torch

    def evaluate(self, board, history: list) -> tuple[list[int], np.ndarray, float]:
        legal = [int(i) for i in self.encoder.legal_action_indices(board)]
        if not legal:
            return [], np.zeros(0, dtype=np.float64), 0.0
        x = self.encoder.encode(board, history)
        prior, value = self.evaluate_many([x], [legal])[0]
        return legal, prior, value

    def evaluate_many(
        self, encodings: list, legal_lists: list[list[int]]
    ) -> list[tuple[np.ndarray, float]]:
        torch = self._torch
        if not encodings:
            return []
        x = np.stack([np.ascontiguousarray(enc) for enc in encodings], axis=0)
        tensor = torch.from_numpy(x).to(self.device)
        ctx = self.lock if self.lock is not None else nullcontext()
        with ctx:
            self.net.eval()
            with torch.no_grad():
                logits, values = self.net(tensor)
        logits = logits.detach().cpu()
        values = values.detach().cpu().reshape(-1)
        out: list[tuple[np.ndarray, float]] = []
        for i, legal in enumerate(legal_lists):
            if not legal:
                out.append((np.zeros(0, dtype=np.float64), float(values[i])))
                continue
            legal_t = torch.tensor(legal, dtype=torch.long)
            prior = torch.softmax(logits[i].index_select(0, legal_t), dim=0)
            out.append((prior.numpy().astype(np.float64), float(values[i])))
        return out


class _Node:
    __slots__ = ("actions", "prior", "n", "w", "child", "expanded", "terminal_v")

    def __init__(self) -> None:
        self.actions: np.ndarray | None = None
        self.prior: np.ndarray | None = None
        self.n: np.ndarray | None = None
        self.w: np.ndarray | None = None
        self.child: list[_Node] | None = None
        self.expanded = False
        self.terminal_v: float | None = None

    def expand(self, actions: list[int], prior: np.ndarray) -> None:
        k = len(actions)
        self.actions = np.asarray(actions, dtype=np.int32)
        self.prior = np.asarray(prior, dtype=np.float64)
        if k and self.prior.sum() > 0:
            self.prior = self.prior / self.prior.sum()
        self.n = np.zeros(k, dtype=np.int32)
        self.w = np.zeros(k, dtype=np.float64)
        self.child = [_Node() for _ in range(k)]
        self.expanded = True

    def mark_terminal(self, value: float) -> None:
        self.terminal_v = float(value)
        self.expanded = True
        self.actions = np.zeros(0, dtype=np.int32)
        self.prior = np.zeros(0, dtype=np.float64)
        self.n = np.zeros(0, dtype=np.int32)
        self.w = np.zeros(0, dtype=np.float64)
        self.child = []

    def select(self, c_puct: float) -> int:
        assert self.n is not None and self.w is not None and self.prior is not None
        n_sum = int(self.n.sum())
        if n_sum == 0:
            return int(np.argmax(self.prior))
        q = np.divide(self.w, self.n, out=np.zeros(self.n.shape, dtype=np.float64), where=self.n > 0)
        u = c_puct * self.prior * math.sqrt(n_sum) / (1.0 + self.n)
        return int(np.argmax(q + u))


@dataclass
class SearchResult:
    move: object
    action_index: int
    policy: list[float]
    visit_counts: dict[int, int]
    root_value: float
    n_simulations: int
    legal_indices: list[int] = field(default_factory=list)


class MCTS:
    def __init__(
        self,
        cfg: Cfg | None = None,
        evaluator: Evaluator | None = None,
        encoder: Encoder | None = None,
        seed: int | None = None,
    ):
        self.cfg = cfg if cfg is not None else load_config()
        if encoder is not None:
            self.encoder = encoder
        elif evaluator is not None and getattr(evaluator, "encoder", None) is not None:
            self.encoder = evaluator.encoder
        else:
            self.encoder = Encoder(self.cfg)
        self.evaluator = evaluator
        mcts = self.cfg["mcts"]
        self.simulations = int(mcts["simulations"])
        self.c_puct = float(mcts["c_puct"])
        self.c_puct_base = float(mcts.get("c_puct_base", 0.0))
        self.c_puct_init = float(mcts.get("c_puct_init", 1.25))
        self.dirichlet_alpha = float(mcts["dirichlet_alpha"])
        self.dirichlet_epsilon = float(mcts["dirichlet_epsilon"])
        self.add_dirichlet_noise = bool(mcts.get("add_dirichlet_noise", True))
        self.temperature = float(mcts["temperature"])
        self.batch_size = max(1, int(mcts.get("batch_size", 1)))
        self.virtual_loss = max(0, int(mcts.get("virtual_loss", 1)))
        self.need_history = int(self.cfg["encode"]["history_length"]) > 1
        rng_seed = self.cfg["seed"] if seed is None else seed
        self.rng = np.random.default_rng(rng_seed)
        self.root = _Node()
        self._ancestors: list[_Node] = []

    def reset(self) -> None:
        self.root = _Node()
        self._ancestors = []

    def ensure_expanded(self, board) -> None:
        if not self.root.expanded:
            self._expand(self.root, board, self.encoder.past_for(board))

    def advance(self, action_index: int) -> bool:
        """Descend into the child of `action_index`. Parent stays reachable via retreat().

        Returns False and leaves the tree unchanged if that edge does not exist.
        """
        if (
            self.root.actions is None
            or self.root.child is None
            or self.root.actions.size == 0
        ):
            return False
        hits = np.flatnonzero(self.root.actions == int(action_index))
        if hits.size == 0:
            return False
        self._ancestors.append(self.root)
        self.root = self.root.child[int(hits[0])]
        return True

    def retreat(self, plies: int = 1) -> bool:
        """Move the root back to an ancestor. Too many plies discards the tree."""
        plies = max(0, int(plies))
        if plies == 0:
            return True
        if plies > len(self._ancestors):
            self.reset()
            return False
        for _ in range(plies):
            self.root = self._ancestors.pop()
        return True

    def _c_puct(self, n_sum: int) -> float:
        if self.c_puct_base > 0:
            return math.log((1.0 + n_sum + self.c_puct_base) / self.c_puct_base) + self.c_puct_init
        return self.c_puct

    def _apply_dirichlet(self, node: _Node) -> None:
        if node.prior is None or node.prior.size == 0:
            return
        noise = self.rng.dirichlet(np.full(node.prior.size, self.dirichlet_alpha, dtype=np.float64))
        eps = self.dirichlet_epsilon
        node.prior = (1.0 - eps) * node.prior + eps * noise

    def _expand(self, node: _Node, board, history: list) -> float:
        tv = terminal_value(board)
        if tv is not None:
            node.mark_terminal(tv)
            return tv
        if self.evaluator is None:
            raise RuntimeError("MCTS needs an evaluator")
        legal, prior, value = self.evaluator.evaluate(board, history)
        if not legal:
            node.mark_terminal(-1.0)
            return -1.0
        node.expand(legal, prior)
        return float(value)

    def _select_leaf(
        self, board, virtual_loss: int
    ) -> tuple[list[tuple[_Node, int]], _Node, list]:
        node = self.root
        path: list[tuple[_Node, int]] = []
        search_hist: list = []
        while node.expanded and node.actions is not None and node.actions.size > 0:
            n_sum = int(node.n.sum()) if node.n is not None else 0
            i = node.select(self._c_puct(n_sum))
            path.append((node, i))
            if virtual_loss:
                node.n[i] += virtual_loss
                node.w[i] -= float(virtual_loss)
            if self.need_history:
                search_hist.append(board.copy())
            self.encoder.play(board, int(node.actions[i]))
            node = node.child[i]
        return path, node, search_hist

    def _backup(self, path: list[tuple[_Node, int]], v: float, virtual_loss: int) -> None:
        for parent, i in reversed(path):
            if virtual_loss:
                parent.n[i] -= virtual_loss
                parent.w[i] += float(virtual_loss)
            v = -v
            parent.n[i] += 1
            parent.w[i] += v

    @staticmethod
    def _unmake(board, path: list) -> None:
        for _ in path:
            board.unmake_move()

    def _simulate(self, board, game_past: list) -> None:
        path, node, search_hist = self._select_leaf(board, virtual_loss=0)
        history = game_past + search_hist
        if not node.expanded:
            v = self._expand(node, board, history)
        else:
            v = 0.0 if node.terminal_v is None else node.terminal_v
        self._backup(path, v, virtual_loss=0)
        self._unmake(board, path)

    def _simulate_batch(self, board, game_past: list, n: int) -> None:
        """Collect n leaves with virtual loss, one batched forward, then backup."""
        vl = self.virtual_loss
        ready: list[tuple[list[tuple[_Node, int]], float]] = []
        pending: list[tuple[list[tuple[_Node, int]], _Node, object, list[int]]] = []
        for _ in range(n):
            path, node, search_hist = self._select_leaf(board, virtual_loss=vl)
            history = game_past + search_hist
            if node.expanded:
                v = 0.0 if node.terminal_v is None else node.terminal_v
                ready.append((path, v))
            else:
                tv = terminal_value(board)
                if tv is not None:
                    node.mark_terminal(tv)
                    ready.append((path, tv))
                else:
                    enc = self.encoder.encode(board, history)
                    legal = [int(i) for i in self.encoder.legal_action_indices(board)]
                    pending.append((path, node, enc, legal))
            self._unmake(board, path)

        id_to_v: dict[int, float] = {}
        unique_nodes: list[_Node] = []
        unique_enc: list = []
        unique_legal: list[list[int]] = []
        seen: dict[int, int] = {}
        for _path, node, enc, legal in pending:
            nid = id(node)
            if nid in seen:
                continue
            seen[nid] = len(unique_nodes)
            unique_nodes.append(node)
            unique_enc.append(enc)
            unique_legal.append(legal)

        if unique_nodes:
            if self.evaluator is None:
                raise RuntimeError("MCTS needs an evaluator")
            pvs = self.evaluator.evaluate_many(unique_enc, unique_legal)
            for node, legal, (prior, value) in zip(unique_nodes, unique_legal, pvs):
                if not node.expanded:
                    if not legal:
                        node.mark_terminal(-1.0)
                        id_to_v[id(node)] = -1.0
                    else:
                        node.expand(legal, prior)
                        id_to_v[id(node)] = float(value)
                elif node.terminal_v is not None:
                    id_to_v[id(node)] = node.terminal_v
                else:
                    id_to_v[id(node)] = float(value)

        for path, v in ready:
            self._backup(path, v, vl)
        for path, node, _enc, _legal in pending:
            self._backup(path, id_to_v[id(node)], vl)

    def run(
        self,
        board,
        simulations: int | None = None,
        add_noise: bool | None = None,
        temperature: float | None = None,
        reuse: bool = False,
    ) -> SearchResult:
        if not reuse:
            self.root = _Node()
        n_sims = self.simulations if simulations is None else int(simulations)
        noise = self.add_dirichlet_noise if add_noise is None else bool(add_noise)
        tau = self.temperature if temperature is None else float(temperature)

        fen_before = board.fen()
        ply_before = board.ply()
        game_past = self.encoder.past_for(board)

        if not self.root.expanded:
            self._expand(self.root, board, game_past)
        if noise and self.root.prior is not None and self.root.prior.size > 1:
            self._apply_dirichlet(self.root)

        if self.root.terminal_v is not None or self.root.actions is None or self.root.actions.size == 0:
            policy = [0.0] * int(self.encoder.action_size)
            return SearchResult(
                move=None,
                action_index=-1,
                policy=policy,
                visit_counts={},
                root_value=0.0 if self.root.terminal_v is None else self.root.terminal_v,
                n_simulations=0,
                legal_indices=[],
            )

        if self.batch_size <= 1 or not hasattr(self.evaluator, "evaluate_many"):
            for _ in range(n_sims):
                self._simulate(board, game_past)
        else:
            done = 0
            while done < n_sims:
                chunk = min(self.batch_size, n_sims - done)
                self._simulate_batch(board, game_past, chunk)
                done += chunk

        if board.fen() != fen_before or board.ply() != ply_before:
            raise RuntimeError("MCTS did not restore the board")

        counts = {int(a): int(n) for a, n in zip(self.root.actions, self.root.n)}
        # Training target π is always N/ΣN. Temperature only chooses the played move.
        policy = _policy_from_counts(self.root.actions, self.root.n, int(self.encoder.action_size))
        action_index = _sample_action(self.root.actions, self.root.n, tau, self.rng)
        move = self.encoder.move_from_index(board, action_index)
        n_sum = float(self.root.n.sum())
        root_value = float(self.root.w.sum() / n_sum) if n_sum else 0.0
        return SearchResult(
            move=move,
            action_index=action_index,
            policy=policy,
            visit_counts=counts,
            root_value=root_value,
            n_simulations=n_sims,
            legal_indices=[int(a) for a in self.root.actions],
        )


def _policy_from_counts(actions: np.ndarray, visits: np.ndarray, size: int) -> list[float]:
    out = [0.0] * size
    if visits.size == 0:
        return out
    total = float(visits.sum())
    if total <= 0:
        p = 1.0 / visits.size
        for a in actions:
            out[int(a)] = p
        return out
    for a, n in zip(actions, visits):
        out[int(a)] = float(n) / total
    return out


def _sample_action(actions: np.ndarray, visits: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    if visits.size == 0:
        raise RuntimeError("no legal actions")
    if temperature <= 1e-8:
        return int(actions[int(np.argmax(visits))])
    p = np.power(visits.astype(np.float64), 1.0 / temperature)
    z = p.sum()
    if z <= 0:
        return int(actions[int(rng.integers(0, actions.size))])
    return int(actions[int(rng.choice(actions.size, p=p / z))])
