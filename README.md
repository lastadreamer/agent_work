# Xiangqi Engine · Phase 1

AlphaZero 式象棋智能体的第一步：**只做规则环境**。没有神经网络，没有 MCTS。

环境错了，自我对弈采到的全是脏数据。这一阶段把走子、将军、终局做成可以独立验证、并且足够快的引擎。

## 这一阶段在学什么

强化学习里的「环境」就是：状态、动作、转移、终止。

| RL 概念 | 这里的实现 |
| --- | --- |
| 状态 \(s\) | `Board`：90 格棋盘 + 轮到谁走 |
| 动作 \(a\) | `Move`：from → to（ICCS，如 `b2e2`） |
| 转移 | `push` / `make_move`（O(1) make-unmake，给以后的 MCTS 用） |
| 终止 | 无棋可走（将死或困毙，走方负）、三次重复、60 回合无进展 |

AlphaZero 论文里的自我对弈循环跑在 C++ 里，Python 只负责训练。本引擎沿用这个分工：

- **C++17 核心**：着法生成、合法性、哈希、perft
- **pybind11**：给之后的 PyTorch 训练循环调用
- **无共享可变状态**：每个线程/进程拿自己的 `Board`（约 16KB，可 `memcpy` / pickle）

## 象棋规则（实现范围）

完整实现：帅/将、仕/士、相/象、马（蹩腿）、车、炮（翻山吃子）、兵/卒（过河后可横走）、九宫、河界、将帅对面、不能送将。

终局：

- **无合法着** → 走方负。在将军中是将死；不在将军中是困毙（和国际象棋不同，困毙算输）。
- **三次重复** → 和棋。这是简化：正式规则还要判长将/长捉，以后需要再接到裁判规则时再加。
- **120 个半回合无吃子、无走兵** → 和棋。

## 构建

依赖：C++17 编译器（建议 `g++`）、`python3-dev`、`pip install pybind11`。

```bash
# Python 模块（训练时用）
python3 -m venv .venv
source .venv/bin/activate
CXX=g++ pip install -e ".[dev]"
pytest -q

# 纯 C++ perft（对照 Chess Programming Wiki）
# 若默认 c++ 是 clang++ 且链不上 libstdc++，指定 g++：
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build build -j
./build/xiangqi-perft check
./build/xiangqi-perft perft 5
```

开局 perft 应对上：depth 1 = 44，depth 2 = 1920，depth 3 = 79666，depth 4 = 3,290,240，depth 5 = 133,312,995。本机 Release 大约 2.2×10⁷ nps。

## 用法

```python
from xiangqi_engine import Board, RED, BLACK, Outcome

b = Board()                 # 开局
for m in b.legal_moves():   # 44 个合法着
    pass
b.push(b.legal_moves()[0])  # 走一步
b.pop()                     # O(1) 悔棋，MCTS 回传用这个
print(b)                    # 中文棋盘
print(b.terminal())         # 是否终局

# 多进程：Board 可 pickle，每个 worker 自己下一盘
from concurrent.futures import ProcessPoolExecutor
from xiangqi_engine import play_random_games
with ProcessPoolExecutor() as pool:
    list(pool.map(play_random_games, [10] * 8, [200] * 8, range(8)))
```

不要在线程之间共享同一个 `Board`。以后自我对弈是「N 个 worker、N 棵搜索树、N 个棋盘」。

## 阶段 2–4：编码 + 策略价值网络

所有可调行为都在 **`config/default.json`**。改通道数、残差块数、历史长度、学习率、MCTS 模拟次数都只动这一份文件。`_` 开头的键是注释，加载时会丢掉。

```python
from xiangqi_engine import Board, Encoder, load_config
from xiangqi_engine.network import build_network, infer

cfg = load_config()          # 或 load_config("config/default.json")
enc = Encoder(cfg)
net = build_network(cfg)
legal, probs, value = infer(net, enc, Board())
```

### 棋盘 → 输入

张量是 `float32`，形状 `(C, 10, 9)`，通道优先。默认 `C = 15`：

| 通道 | 含义 |
| --- | --- |
| 0–6 | 当前走方的 帅仕相马车炮兵 |
| 7–13 | 对方的 7 类棋子 |
| 14 | `halfmove / 120`（无进展计数，整盘常数平面） |

`encode.perspective = current_player` 时，**黑方走则把棋盘转 180°**（格子 `sq' = 89 - sq`），所以网络永远看见「我在 rank 0」。价值头的 `v ∈ [-1, 1]` 也是**当前走方**的胜负期望，不是永远从红方看。

`history_length > 1` 时，每个历史局面再叠 14 个棋子平面，全部按**现在**的走方当「我」。这和 AlphaZero 一致。

### 输出 → 动作

策略头输出 **8100** 维 logits：`index = from * 90 + to`，格子已经是网络坐标系（黑方走时已翻转）。非法着在 softmax 前 mask 掉。落子时用 `index_to_move(index, flip=黑方)` 变回引擎着法。

价值头：`tanh`，标量 `v`。

网络是论文里的双头残差塔：`Conv → N × ResBlock → {policy 1×1+FC, value 1×1+FC+tanh}`。块数和通道在 `network.*`。

## 阶段 5：MCTS

搜索在 Python 里，走子/悔棋仍走 C++。一棵树对应一个 `Board`，不要跨线程共享。网络可以共享，但 `NetworkEvaluator` 要加锁。

```python
from xiangqi_engine import Board, Encoder, load_config
from xiangqi_engine.mcts import MCTS, UniformEvaluator, NetworkEvaluator

cfg = load_config()
enc = Encoder(cfg)
mcts = MCTS(cfg, UniformEvaluator(enc))          # 测搜索结构
# mcts = MCTS(cfg, NetworkEvaluator(net, enc))   # 接上策略价值网
result = mcts.run(Board(), add_noise=True, temperature=1.0)
board = Board()
board.push(result.move)
# result.policy 是 8100 维的 π，给以后的训练用
```

每条模拟：用 PUCT 选到叶子 → 网络给出 \(p,v\)（终局用胜负，不调用网络）→ 沿路回传，**每一层把 \(v\) 取反**（零和）。根节点加 Dirichlet 噪声。`result.policy` 永远是 \(N/\sum N\)（训练标签 \(\pi\)）。温度只决定走出哪步：\(\tau=1\) 按 \(N^{1/\tau}\) 采样，\(\tau=0\) 取访问最多的着。

超参数在 JSON 的 `mcts` 段：`simulations`、`c_puct`、`dirichlet_*`、`temperature`。

## 阶段 6–8：自我对弈、训练、评估闭环

一盘自我对弈在每个决策点存 \((s, \pi)\)，终局得到 \(z\in\{+1,0,-1\}\)（**当前走方**视角：红胜则红方局面 \(z=+1\)、黑方局面 \(z=-1\)）。回放池按 `replay.capacity` 环形保存；\(\pi\) 只存合法着上的质量。

训练一步：

\[
L = w_p\,\mathrm{CE}(\pi, p) + w_v\,(z-v)^2
\]

权重衰减在优化器里。评估时不开 Dirichlet、温度 0；挑战者得分（胜=1、和=0.5）达到 `eval.win_rate_threshold` 就晋升为 best。

```bash
# 完整超参数：config/default.json
# 冒烟（小网络、少模拟）：
python -m xiangqi_engine.loop --config config/smoke.json
```

```python
from xiangqi_engine.selfplay import play_game
from xiangqi_engine.replay import ReplayBuffer
from xiangqi_engine.train import train_batches

rec = play_game(cfg, evaluator, encoder, seed=0)
buffer.extend(rec.samples)
train_batches(net, buffer, cfg)
```

多进程采样：`selfplay.n_workers > 1` 时每个进程自己的 `Board` / 编码器 / 搜索树，网络用 CPU `state_dict` 拷贝。不要在进程间共享一块棋盘。

所有旋钮仍在 JSON：`selfplay`、`replay`、`train`、`eval`、`loop`。
