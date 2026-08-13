# 中国象棋 AlphaZero 智能体

用 AlphaZero 的方式下中国象棋：C++ 规则引擎负责走子和终局，Python 负责策略价值网络、MCTS、自我对弈训练，以及本地网页对弈。

克隆后可以做三件事：

- **训练**：自我对弈采样 → 更新网络 → 评估，循环写出权重
- **对弈**：浏览器里人人 / 人机 / 机机，支持悔棋
- **当引擎用**：在 Python 里生成合法着、走子、悔棋、判断终局

超参集中在一份 JSON：`config/default.json`。`_` 开头的键是注释，加载时会丢掉。

## 环境配置

需要：

- Python 3.9+
- C++17 编译器（建议 `g++`）
- 对应版本的 Python 头文件（Debian/Ubuntu：`python3-dev`）
- 训练和对弈里的引擎走棋还需要 **PyTorch**

本仓库用 pybind11 编译 C++ 扩展。若系统默认 `c++` 是 clang++ 且链不上 `libstdc++`，安装和编译时指定 `CXX=g++`。

```bash
git clone https://github.com/lastadreamer/agent_work.git
cd agent_work

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

CXX=g++ pip install -e ".[dev]"    # 含 pytest、torch；只要训练可用 ".[train]"
pytest -q
```

装好后有两个命令：`xiangqi-train`、`xiangqi-play`。也可以用 `python -m xiangqi_engine.loop` / `python -m xiangqi_engine.play`。

可选：单独编一个 C++ perft，用来对照 [Chess Programming Wiki](https://www.chessprogramming.org/Chinese_Chess) 的节点数。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build build -j
./build/xiangqi-perft check
```

开局应对上：depth 1 = 44，2 = 1920，3 = 79666，4 = 3,290,240，5 = 133,312,995。

## 训练

先跑冒烟配置，确认闭环能转：小网络、少模拟、一局迭代很快结束。

```bash
python -m xiangqi_engine.loop --config config/smoke.json
# 或：xiangqi-train --config config/smoke.json
```

正式训练读 `config/default.json`（也可设环境变量 `XIANGQI_CONFIG` 指向自己的文件）：

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

**一盘棋里**会留下已探索的子树：走出一步后，把该着对应的子树提成新根，下一步搜索接着用上面的 \(N,Q,P\)（`mcts.reuse_tree`，默认开）。**不会**把树写进训练快照。换一盘棋、或网络已经更新之后，旧树的统计量对不上新局面/新先验，存下来没有用。只拿到旧的 `best.pt`（没有 sidecar 回放、没有优化器）也能接着训，但回放池是空的、优化器会重开。

`default.json` 按「能认真训」给的：每轮 64 盘、每步 400 次模拟、6×128 网络。机器吃不消就先改小 `selfplay.n_games_per_iter`、`mcts.simulations`、`network.blocks` / `channels`，或继续用 `smoke.json` 当模板另存一份。

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

悔一步撤 1 个半回合；悔一回合在人机里会连着撤回引擎的应着，直到又轮到人。权重路径留空则用均匀先验 MCTS，不需要已经训好的网络。

`config/default.json` 的 `play` 段只给这个界面用（主机、端口、默认红黑角色、对弈模拟次数）。训练仍读 `mcts.simulations`。

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
- **同一局面出现三次** → 和棋
- **120 个半回合无吃子、无走兵** → 和棋

正式规则里的长将、长捉还没做。现在长将可以被三次重复判和，训练标签会因此脏一截。

## 目录

```
config/                 default.json、smoke.json
include/xiangqi/        C++ 头文件
src/                    规则、编码、pybind11、perft
python/xiangqi_engine/  配置、编码、网络、MCTS、训练、对弈 UI
tests/
```
