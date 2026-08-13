# 中国象棋 AlphaZero 智能体

用 AlphaZero 的方式下中国象棋：C++ 规则引擎负责走子和终局，Python 负责策略价值网络、MCTS、自我对弈训练，以及本地网页对弈。

克隆后可以做三件事：

- **训练**：自我对弈采样 → 更新网络 → 评估，循环写出权重
- **对弈**：浏览器里人人 / 人机 / 机机，支持悔棋
- **当引擎用**：在 Python 里生成合法着、走子、悔棋、判断终局

超参集中在一份 JSON：`config/default.json`。`_` 开头的键是注释，加载时会丢掉。

## 环境配置

两套 extra，用 `uv.lock` 对齐版本。核心引擎（numpy + C++ 扩展）始终安装；PyTorch / pytest 按用途选：

| extra | 给谁用 | 能做什么 |
| --- | --- | --- |
| **`test`** | Mac / 本机 | 单测、CPU 冒烟、网页对弈、加载 `best.pt` 人机/机机 |
| **`train`** | H200 等训练机 | 上面这些，再加上按 `default.json` 正式自我对弈训练 |

需要：

- [uv](https://docs.astral.sh/uv/)
- Python 3.10+（仓库钉的是 3.12，没有的话 `uv` 会按 `.python-version` 装）
- C++17 编译器（Linux 建议 `g++`；macOS 用自带 clang 即可）
- Debian/Ubuntu 还要 `python3-dev`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/lastadreamer/agent_work.git
cd agent_work
```

**本机 / Mac（test）**

```bash
uv sync --extra test
# Linux 若 clang++ 链不上 libstdc++：CXX=g++ uv sync --extra test

uv run pytest -q
uv run xiangqi-train --config config/smoke.json
uv run xiangqi-play --checkpoint checkpoints/best.pt
```

**训练机（train）**

```bash
CXX=g++ uv sync --extra train
uv run xiangqi-train
```

`uv sync` 之后也可以 `source .venv/bin/activate`，直接用 `pytest`、`xiangqi-train`、`xiangqi-play`。改过 `pyproject.toml` 后重新 `uv lock`，把 `uv.lock` 一并提交。

没有 uv 时：

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"    # 或 ".[train]"
```

可选：单独编一个 C++ perft，用来对照 [Chess Programming Wiki](https://www.chessprogramming.org/Chinese_Chess) 的节点数。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build build -j
./build/xiangqi-perft check
```

开局应对上：depth 1 = 44，2 = 1920，3 = 79666，4 = 3,290,240，5 = 133,312,995。

## 训练

本机或 Mac 上先跑冒烟：强制 CPU、小网络、少模拟、单进程，不需要 CUDA。

```bash
python -m xiangqi_engine.loop --config config/smoke.json
# 或：xiangqi-train --config config/smoke.json
pytest -q
```

正式训练读 `config/default.json`（`device` / `selfplay.device` 为 `auto`：有 CUDA 用 GPU，没有就用 CPU）。H200 上会走到 GPU；Mac 上误跑这份也会落到 CPU，只是 20×256 + 256 盘会极慢，所以本机请用 `smoke.json`。也可设 `XIANGQI_CONFIG` 指向自己的文件：

```bash
python -m xiangqi_engine.loop
xiangqi-train --iterations 20
```

每一轮大致是：当前网络自我对弈 → 样本进回放池 → 梯度更新 → 和 best 对打，胜率够了再晋升。日志写在 `logs/train.jsonl`。快照会写：

| 文件 | 内容 |
| --- | --- |
| `checkpoints/iter_XXXX.pt` + `iter_XXXX.replay.pkl` | 该轮完整断点：当前网、best 网、优化器、迭代号、回放池 |
| `checkpoints/latest.pt` + `latest.replay.pkl` | 同上，始终指向最近一次保存 |
| `checkpoints/best.pt` | 仅晋升后的权重，给对弈界面用 |

中断之后从断点继续（`--iterations` 表示**再跑多少轮**，不是从头数到第几轮）：

```bash
xiangqi-train --resume                          # 读 latest.pt
xiangqi-train --checkpoint checkpoints/iter_0005.pt --iterations 20
```

会恢复：网络权重、当时的 best、Adam/SGD 状态、回放池、迭代序号。下一轮自我对弈的随机种子仍由 `seed + iteration` 决定，所以不会和已经跑过的轮次撞车。

**一盘棋里**（自我对弈、评估、网页对弈）共用一棵树：走出一步后降到该着的子节点，悔棋回到父节点再搜（`mcts.reuse_tree`，默认开）。**不会**把树写进训练快照。换一盘棋、或网络已经更新之后，旧树的统计量对不上新局面/新先验，存下来没有用。只拿到旧的 `best.pt`（没有 sidecar 回放、没有优化器）也能接着训，但回放池是空的、优化器会重开。

`default.json` 按 **4×H200** 写的：20×256 网络、每步 800 次模拟、每轮 256 盘 / 32 个 worker（推理摊到 `selfplay.gpus` 张卡）、训练 batch 4096。梯度更新只占一张卡；自我对弈才是瓶颈。你贴的 `nvidia-smi` 里四张卡已经各占 110GB+，先空出卡再训，或：

```bash
CUDA_VISIBLE_DEVICES=0,1 xiangqi-train
# 同时把 config 里 selfplay.gpus 改成实际空闲张数
```

机器吃不消就改小 `selfplay.n_games_per_iter`、`n_workers`、`mcts.simulations`，或先跑 `config/smoke.json`。

没有训过的网络加上很少的模拟，棋力接近乱走，这是预期现象。

## 对弈

```bash
python -m xiangqi_engine.play
# 指定权重和思考量：
xiangqi-play --checkpoint checkpoints/best.pt --simulations 200
```

浏览器打开提示的地址，默认 `http://127.0.0.1:8765`。

| 模式 | 行为 |
| --- | --- |
| 人人 | 双方点棋 |
| 人机（执红 / 执黑） | 轮到引擎自动走 |
| 机机 | 「机机自动」连走，或「引擎走一步」 |

悔一步撤 1 个半回合；悔一回合在人机里会连着撤回引擎的应着，直到又轮到人。一局棋共用一棵 MCTS：走子把当前节点降到对应子节点，悔棋回到父节点再搜，已经走过的分支还在。权重路径留空则用均匀先验 MCTS，不需要已经训好的网络。

`config/default.json` 的 `play` 段只给这个界面用（主机、端口、默认红黑角色、对弈模拟次数）。训练仍读 `mcts.simulations`。

## 五子棋

和象棋共用同一套自我对弈 / 训练 / 评估循环，棋盘和着法分开。自由五子（五连即胜，无禁手），黑先。默认象棋路径不变。

```bash
# 本机冒烟
python -m xiangqi_engine.loop --config config/gomoku_smoke.json

# 正式五子棋（15×15，比象棋快，用来确认算法在学）
python -m xiangqi_engine.loop --config config/gomoku.json

# 对弈，默认端口 8766
python -m xiangqi_engine.play --config config/gomoku.json
```

权重和日志写在 `checkpoints/gomoku`、`logs/gomoku`，不会覆盖象棋的 `checkpoints/best.pt`。

## 配置

改行为优先改 JSON，不要改代码。常用段：

| 段 | 作用 |
| --- | --- |
| `device` | `auto` / `cpu` / `cuda` |
| `paths` | 权重、回放、日志、`latest_checkpoint` |
| `network` | 残差块数、通道 |
| `mcts` | 模拟次数、PUCT、温度、Dirichlet |
| `selfplay` | 每轮盘数、进程数、单局最长步数 |
| `replay` / `train` | 回放池容量、优化器、学习率、batch |
| `eval` / `loop` | 评估盘数、晋升阈值、迭代次数 |
| `play` | 网页对弈；不影响训练 |

覆盖方式：`--config 路径`，或 `export XIANGQI_CONFIG=/path/to/your.json`。

## 在代码里用引擎

不要在线程或进程之间共享同一块 `Board`。每个 worker 自己一盘棋、一棵搜索树。

```python
from xiangqi_engine import Board, Encoder, load_config
from xiangqi_engine.mcts import MCTS, UniformEvaluator

b = Board()                 # 开局
for m in b.legal_moves():   # 合法着，ICCS 如 b2e2
    pass
b.push(b.legal_moves()[0])
b.pop()                     # O(1) 悔棋
print(b.fen(), b.terminal())

cfg = load_config()
enc = Encoder(cfg)
result = MCTS(cfg, UniformEvaluator(enc)).run(Board(), add_noise=False, temperature=0.0)
```

网络和完整训练循环：

```python
from xiangqi_engine.network import build_network, infer
from xiangqi_engine.selfplay import play_game
from xiangqi_engine.replay import ReplayBuffer
from xiangqi_engine.train import train_batches

net = build_network(cfg)
legal, probs, value = infer(net, enc, Board())
```

## 规则

走子按中国象棋常见实现：帅/将、仕/士、相/象、马（蹩腿）、车、炮、兵/卒（过河后可横走）、九宫、河界、将帅对面、不能送将。

终局：

- **无合法着** → 走方负（将死或困毙；困毙算输，和国际象棋不同）
- **同一局面出现三次** → 看最近一个循环：单方长将负，单方长捉（或一将一捉）负；双方都犯规或双方都是闲着 → 和（允许不变）
- **120 个半回合无吃子、无走兵** → 和棋

长将、长捉只在三次重复时裁决，不改变着法合法性，也不改 perft。捉的计算机可判定子集：攻击无根的非将/帅子；未过河兵/卒不算被捉；仅兵/卒去捉不算（兵允许长捉）。有根 = 被己方攻击。长杀还没做。

## 目录

```
config/                 default.json、smoke.json
include/xiangqi/        C++ 头文件
src/                    规则、编码、pybind11、perft
python/xiangqi_engine/  配置、编码、网络、MCTS、训练、对弈 UI
tests/
```
