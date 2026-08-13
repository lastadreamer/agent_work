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

## 下一步（还没做）

等你点头后再做阶段 2：把 `Board` 编成神经网络输入张量（棋子平面）。
