# AlphaZero 棋类智能体

用 AlphaZero 的方式下棋：自我对弈采样 → 更新策略价值网络 → 评估晋升。目前两套棋共用同一条训练和对弈管线，用 JSON 切换。

| | 中国象棋 | 五子棋 |
| --- | --- | --- |
| 配置 | `config/default.json`（正式）、`config/smoke.json`（冒烟） | `config/gomoku.json`（正式）、`config/gomoku_smoke.json`（冒烟） |
| 规则引擎 | C++（走子、将军、长将/长捉） | Python（自由五子，无禁手） |
| 棋盘 / 动作 | 10×9，8100 个 from-to | 15×15（冒烟 9×9），落子 |
| 对弈页 | http://127.0.0.1:8765 | http://127.0.0.1:8766 |
| 权重目录 | `checkpoints/` | `checkpoints/gomoku/` |

克隆后可以做三件事：

- **训练**：冻结的 best 自我对弈 → 回放池 → 更新学习者 → 和 best 对打，赢了才晋升
- **对弈**：浏览器里人人 / 人机 / 机机，支持悔棋
- **当引擎用**：在 Python 里生成合法着、走子、悔棋、判断终局

超参都在 JSON 里。`_` 开头的键是注释，加载时会丢掉。改行为优先改配置，不要改代码。

## 环境

两套 extra，用 `uv.lock` 对齐版本。核心引擎（numpy + C++ 扩展）始终安装；PyTorch / pytest 按用途选：

| extra | 给谁用 | 能做什么 |
| --- | --- | --- |
| **`test`** | Mac / 本机 | 单测、CPU 冒烟、网页对弈、加载已有权重 |
| **`train`** | H200 等训练机 | 上面这些，再加上按正式配置自我对弈 |

需要：[uv](https://docs.astral.sh/uv/)、Python 3.10+（仓库钉 3.12）、C++17 编译器（Linux 用 `g++`，macOS 用自带 clang）、Debian/Ubuntu 还要 `python3-dev`。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/lastadreamer/agent_work.git
cd agent_work
```

**本机 / Mac**

```bash
uv sync --extra test
# Linux 若 clang++ 链不上 libstdc++：CXX=g++ uv sync --extra test

uv run pytest -q
uv run xiangqi-train --config config/smoke.json
uv run xiangqi-play --checkpoint checkpoints/best.pt
```

**训练机**

```bash
CXX=g++ uv sync --extra train
uv run xiangqi-train                          # 象棋，读 default.json
uv run xiangqi-train --config config/gomoku.json
```

`uv sync` 之后也可以 `source .venv/bin/activate`，直接用 `pytest`、`xiangqi-train`、`xiangqi-play`。只改了 C++ 源码、版本号没变时，`uv sync` 可能不重编，需要：

```bash
uv sync --extra test --reinstall-package xiangqi-engine
```

没有 uv 时：`python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[test]"`（或 `".[train]"`）。

可选：单独编 C++ perft，对照 [Chess Programming Wiki](https://www.chessprogramming.org/Chinese_Chess) 的节点数。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build build -j
./build/xiangqi-perft check
```

开局应对：depth 1 = 44，2 = 1920，3 = 79666，4 = 3,290,240，5 = 133,312,995。

## 训练

同一套循环，换配置即换棋种。每一轮：用冻结的 **best** 自我对弈 → 样本进回放池 → 更新学习者 → 和 best 对打，胜率达到 `eval.win_rate_threshold` 才把学习者写进 best，下一轮才用新权重造棋。终端会打进度和 ETA；明细在对应的 `logs/**/train.jsonl`。

本机先冒烟（强制 CPU、小网络、少模拟，不需要 CUDA）：

```bash
python -m xiangqi_engine.loop --config config/smoke.json
python -m xiangqi_engine.loop --config config/gomoku_smoke.json
```

正式训练（`device` / `selfplay.device` 为 `auto`：有 CUDA 用 GPU，没有就 CPU）：

```bash
# 象棋：20×256，每步 800 次模拟，每轮 256 盘 / 32 worker
xiangqi-train
# 或：python -m xiangqi_engine.loop

# 五子棋：15×15，6×64，每步 200 次模拟，每轮 128 盘；比象棋快，适合先看算法是否在变强
xiangqi-train --config config/gomoku.json
```

Mac 上不要跑 `default.json` / `gomoku.json` 的正式规模。只跑指定轮数：`xiangqi-train --iterations 20`。也可 `export XIANGQI_CONFIG=/path/to/your.json`。

快照（路径由配置的 `paths` 决定；五子棋在 `checkpoints/gomoku/`，不会覆盖象棋）：

| 文件 | 内容 |
| --- | --- |
| `iter_XXXX.pt` + `iter_XXXX.replay.pkl` | 该轮完整断点：当前网、best、优化器、迭代号、回放池 |
| `latest.pt` + `latest.replay.pkl` | 同上，始终指向最近一次保存 |
| `best.pt` | 当前门控权重：自我对弈和对弈界面都用它 |

中断后续训（`--iterations` 表示**再跑多少轮**）：

```bash
xiangqi-train --resume
xiangqi-train --checkpoint checkpoints/iter_0005.pt --iterations 20
xiangqi-train --config config/gomoku.json --resume
```

会恢复网络、当时的 best、优化器、回放池、迭代序号。自我对弈种子仍是 `seed + iteration`，不会和已跑过的轮次撞车。

**一盘棋里**（自我对弈、评估、网页对弈）共用一棵 MCTS：走出一步降到子节点，悔棋回到父节点（`mcts.reuse_tree`，默认开）。树不写进训练快照。只拿到旧的 `best.pt` 也能接着训，但回放是空的、优化器会重开。

`default.json` 按 **4×H200** 写的。梯度更新占一张卡，自我对弈才是瓶颈。卡被占满时先空出来，或：

```bash
CUDA_VISIBLE_DEVICES=0,1 xiangqi-train
# 同时把 config 里 selfplay.gpus 改成实际空闲张数
```

自我对弈里每个 worker 会把最多 `mcts.batch_size` 个叶子一次送给 GPU（默认 32），用 `virtual_loss` 让同一批里的 PUCT 不会全挤在同一条边。机器吃不消就改小 `selfplay.n_games_per_iter`、`n_workers`、`mcts.simulations`。没有训过的网络加上很少的模拟，棋力接近乱走，这是预期现象。

## 对弈

```bash
# 象棋，默认 http://127.0.0.1:8765
python -m xiangqi_engine.play
xiangqi-play --checkpoint checkpoints/best.pt --simulations 200

# 五子棋，默认 http://127.0.0.1:8766
python -m xiangqi_engine.play --config config/gomoku.json
xiangqi-play --config config/gomoku.json --checkpoint checkpoints/gomoku/best.pt
```

浏览器打开终端里打印的地址。棋盘按窗口缩放并居中。

| 模式 | 行为 |
| --- | --- |
| 人人 | 双方点棋（象棋点从–到，五子棋点交叉点） |
| 人机 | 轮到引擎自动走 |
| 机机 | 「机机自动」连走，或「引擎走一步」 |

悔一步撤 1 个半回合；悔一回合在人机里会连着撤回引擎的应着。权重路径留空则用均匀先验 MCTS，不需要已经训好的网络。`play` 段只给界面用，训练仍读 `mcts.simulations`。

## 配置

| 段 | 作用 |
| --- | --- |
| `game` | `xiangqi`（默认）或 `gomoku` |
| `device` | `auto` / `cpu` / `cuda` |
| `paths` | 权重、回放、日志、`latest_checkpoint` |
| `board` / `action` | 棋盘大小、动作编码与数量 |
| `network` | 残差块数、通道 |
| `mcts` | 模拟次数、PUCT、温度、Dirichlet、叶子评估 `batch_size` |
| `selfplay` | 每轮盘数、进程数、单局最长步数 |
| `replay` / `train` | 回放池容量、优化器、学习率、batch |
| `eval` / `loop` | 评估盘数、晋升阈值、迭代次数 |
| `play` | 网页对弈；不影响训练 |

覆盖：`--config 路径`，或 `export XIANGQI_CONFIG=/path/to/your.json`。

## 在代码里用

不要在线程或进程之间共享同一块棋盘。每个 worker 自己一盘棋、一棵搜索树。

```python
from xiangqi_engine import Board, Encoder, load_config
from xiangqi_engine.game import make_board, make_encoder
from xiangqi_engine.mcts import MCTS, UniformEvaluator

# 象棋
b = Board()
for m in b.legal_moves():   # ICCS，如 b2e2
    pass
b.push(b.legal_moves()[0])
b.pop()
print(b.fen(), b.terminal())

cfg = load_config()
enc = Encoder(cfg)
result = MCTS(cfg, UniformEvaluator(enc)).run(Board(), add_noise=False, temperature=0.0)

# 五子棋（或任何一份带 game 字段的配置）
gcfg = load_config("config/gomoku.json")
gb = make_board(gcfg)
genc = make_encoder(gcfg)
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

**象棋。** 帅/将、仕/士、相/象、马（蹩腿）、车、炮、兵/卒（过河后可横走）、九宫、河界、将帅对面、不能送将。终局：

- 无合法着 → 走方负（将死或困毙；困毙算输）
- 同一局面出现三次 → 看最近一个循环：单方长将负，单方长捉（或一将一捉）负；双方都犯规或都是闲着 → 和
- 120 个半回合无吃子、无走兵 → 和

长将、长捉只在三次重复时裁决，不改变着法合法性，也不改 perft。捉的计算机可判定子集：攻击无根的非将/帅；未过河兵/卒不算被捉；仅兵/卒去捉不算。有根 = 被己方攻击。长杀还没做。

**五子棋。** 自由五子，黑先，横竖斜五连即胜，满盘和。没有连珠禁手。

## 目录

```
config/                 default / smoke / gomoku / gomoku_smoke
include/xiangqi/        象棋 C++ 头文件
src/                    象棋规则、编码、pybind11、perft
python/xiangqi_engine/  配置、网络、MCTS、训练、象棋/五子棋对弈 UI
python/xiangqi_engine/gomoku/   五子棋棋盘与编码
tests/
```
