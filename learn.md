# 学习笔记：这个仓库里的 AlphaZero 是怎么实现的

这份文档对着**当前源码**写，目的是让你（以及以后读这份代码的人）能把「论文里的 AlphaZero」和「这个仓库实际做了什么、为什么这么做、训练时踩过什么坑」对上。README 讲怎么跑；这里讲每一块怎么接、数字从哪来、哪些地方和论文/象棋规则有意简化。

仓库包名 `xiangqi-engine` 0.3.0。象棋规则在 C++17 + pybind11；网络 / MCTS / 训练 / 网页对弈在 Python。五子棋规则全是 Python，和象棋共用同一条训练环。

读的时候建议：先看第 1–2 节把循环画出来，再按你关心的子系统往下翻。训练经验单独放在第 16 节，那是真机跑出来的，不是纸上推演。

---

## 1. 这个项目在学什么

目标不是冲国际象棋/日本将棋那种算力规模，而是把 AlphaZero 的闭环在**中国象棋**上跑通，棋力先够赢家里人。五子棋配置是验证「算法会不会变强」的探针：盘面简单、终局早、同样的 `loop → selfplay → train → eval`，几小时就能看出策略损失和评估胜率有没有动。

论文对照（实现时要分清两篇）：

| 来源 | 自我对弈用谁 | 本仓库 |
| --- | --- | --- |
| AlphaGo Zero (2017) | 冻结的 **best**，学习者打赢才晋升 | **现在用这个**（`loop.py` 里 `gate=best`） |
| AlphaZero (2018) | 始终用**最新**网络，没有门控 | 象棋早期实验用过，小算力下会死亡螺旋，见 §16 |

其余结构更接近 2018 象棋/将棋论文：残差双头网、PUCT、Dirichlet 开局噪声、温度只用来采样着法、训练目标 π 永远是访问次数归一化、损失函数是策略交叉熵 + 价值 MSE。

数据流（每一轮 iteration）：

```
冻结的 best ──自我对弈──► (s, π, z) 进 ReplayBuffer
                              │
学习者 net ◄──Adam 更新── train_batches
    │
    └─ eval：net vs best（无 Dirichlet，τ=0）
         win_rate = (胜 + 0.5×和) / 盘数
         ≥ eval.win_rate_threshold（默认 0.55）才 best ← net
```

自我对弈**从来不读学习者**。学习者再花哨，下一轮造棋的仍是晋升过的 best。第一轮开始时 `best` 和 `net` 是同一份随机初始化。

---

## 2. 仓库地图与构建

```
config/                 JSON 超参。`_` 开头的键是注释，加载时丢掉。
include/xiangqi/        C++ 头：board.hpp（规则）、encode.hpp（平面与动作）
src/board.cpp           走子、将军、长将/长捉、终局、perft
src/encode.cpp          CHW 编码 + legal_indices
src/bindings.cpp        pybind11 模块 xiangqi_engine._xiangqi
src/perft_main.cpp      独立 perft 可执行文件（CMake）
python/xiangqi_engine/  Python 管线
  game.py               按 cfg["game"] 选棋盘/编码器
  config.py             加载、校验、EncodeSpec、device
  encode.py             象棋 Encoder（历史窗 + 动作索引）
  gomoku/board.py       自由五子棋盘
  gomoku/encode.py      五子棋平面
  network.py            残差双头网（import torch；不要在 __init__.py 里 import）
  mcts.py               PUCT、批叶子、虚拟损失、树复用
  selfplay.py           一盘棋 → 样本；多进程 worker
  replay.py             稀疏 π 的 deque
  train.py              CE + MSE + Adam
  evaluate.py           两评估器对打
  loop.py               一轮完整循环 + 断点
  play/                 本地 HTTP 对弈
tests/                  pytest
```

入口：

| 命令 | 实际函数 |
| --- | --- |
| `xiangqi-train` / `python -m xiangqi_engine` / `python -m xiangqi_engine.loop` | `loop.main` |
| `xiangqi-play` / `python -m xiangqi_engine.play` | `play.server.main` |
| `xiangqi-plot` / `python -m xiangqi_engine.plot_logs` | `plot_logs.main`（读 jsonl 画曲线，不 import torch） |

**不要在 `xiangqi_engine/__init__.py` 里 import torch。** 那个文件只 re-export C++ 引擎、Encoder、MCTS、UniformEvaluator。对弈 UI、只加载规则的脚本因此不必把 CUDA 运行时拉起来。网络在 `network.py` / `loop.py` / `selfplay` worker 里按需 import。

构建：`setup.py` 用 pybind11 编 `xiangqi_engine._xiangqi`，源文件 `board.cpp` + `encode.cpp` + `bindings.cpp`，C++17，`-O3 -DNDEBUG`。`pyproject.toml` 核心依赖只有 numpy；`test` extra 拉 pytest + torch；`train` extra 再加 matplotlib（画 `train.jsonl`）。

Linux 上 `/usr/bin/c++` 经常是 clang++，链 `libstdc++` 会失败。训练机/CI 用：

```bash
CXX=g++ uv sync --extra train   # 或 --extra test
```

Mac 用系统 clang。只改了 C++、版本号没变时 `uv sync` 可能不重编，需要 `--reinstall-package xiangqi-engine`。

独立 perft（对照 [CPW Chinese Chess Perft](https://www.chessprogramming.org/Chinese_Chess_Perft_Results)）：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build build -j
./build/xiangqi-perft check
```

开局：depth 1 = 44，2 = 1920，3 = 79666，4 = 3,290,240，5 = 133,312,995。长将/长捉**只在三次重复时裁决**，不改变着法合法性，所以 perft 数字不受影响。

线程模型：`Board` 是 trivially copyable（`static_assert`），可以 memcpy。**不要在线程/进程之间共享同一块棋盘或同一棵 MCTS。** 每个 worker 自己 `Board`、自己 `Encoder`、自己 `MCTS`。网络可以共享，但 `NetworkEvaluator` 要带 lock；当前自我对弈是**一进程一网**，不用锁。

颜色约定（两套棋共用 C++ 枚举）：`RED = 0`，`BLACK = 1`。象棋红先。五子棋黑先，白方复用 `RED`（所以评估里「红胜」在五子棋 UI 上显示成「白胜」）。

---

## 3. 配置系统

所有行为优先改 JSON，不要改代码。加载：`load_config(path)`；不传路径则看环境变量 `XIANGQI_CONFIG`，再试 `cwd/config/default.json`、仓库根下的 `config/default.json`。

`Cfg` 是带属性访问的 dict：`cfg.network.blocks` 和 `cfg["network"]["blocks"]` 都行。`_strip_doc` 递归丢掉键名以 `_` 开头的项（包括 `"_doc"`），所以注释可以写在 JSON 里而不进校验。

`validate_config`：

- `encode.perspective` 只能是 `current_player` 或 `absolute`
- `history_length >= 1`
- 网络各尺寸 ≥ 1
- 象棋：`action.encoding == "from_to"` 且 `action.size == 8100`（`ACTION_FROM_TO = 90×90`）
- 五子棋：`game == "gomoku"`，棋盘必须正方形且边长 ≥ 5，`action.encoding == "place"`，`action.size == ranks * files`

`device` / `selfplay.device`：`auto` 表示有 CUDA 用 GPU，没有（Mac、CI）退回 CPU。`resolve_device` 在 import torch 失败时也返回 `cpu`。

四份配置：

| 文件 | 用途 |
| --- | --- |
| `config/default.json` | 象棋正式：20×256，800 sims，按 4×H200 写 |
| `config/smoke.json` | 象棋冒烟：1×16，8 sims，强制 CPU |
| `config/gomoku.json` | 五子棋正式：6×64，200 sims（仓库默认；训练时你改过，见 §16） |
| `config/gomoku_smoke.json` | 五子棋冒烟 |

`default.json` 的 `rules` 段（三次重复、120 半回合、困毙算输）是**文档**，C++ 引擎里对应常量是写死的，改 JSON **不会**改规则。两边必须靠人和测试保持一致。

`loop.n_iterations` 是「这一次命令再跑多少轮」，不是「一共训到第几轮」。`--resume` 从 `latest.pt` 读出已完成的 iteration，再往后跑 `n_iterations` 次。`--iterations 20` 覆盖这个字段。

---

## 4. 游戏工厂

`python/xiangqi_engine/game.py` 只有三个函数：

- `game_name(cfg)`：缺省或空 → `"xiangqi"`
- `make_board(cfg)`：gomoku → `GomokuBoard(files)`，否则 C++ `Board()`
- `make_encoder(cfg)`：gomoku → `GomokuEncoder`，否则 `Encoder`

自我对弈、评估、训练循环、网页对弈都走工厂，所以换棋种 = 换 JSON 的 `game` 字段（外加棋盘/动作/网络尺寸）。MCTS 本身不 if-game：它只认 encoder 的 `encode` / `play` / `legal_action_indices` / `move_from_index` / `action_size`，以及 board 的 `terminal` / `side_to_move` / `make_move` / `unmake_move` / `fen` / `ply`。

---

## 5. 象棋 C++ 引擎

头文件 `include/xiangqi/board.hpp`，实现 `src/board.cpp`。这是整条管线的地面真相：合法着、终局、重复。

### 5.1 坐标、棋子、FEN

- 10 行 × 9 列 = 90 格。`sq = rank * 9 + file`。
- **rank 0 是红方底线**，rank 9 是黑方底线。file 0 是红方视角最左。
- ICCS：`a0`–`i9`。着法四个字符，如 `b2e2`（炮从 b2 到 e2）。
- 棋子 1–7 红（帅仕相马车炮兵），8–14 黑（将士象馬車砲卒）。0 空。
- 开局 FEN 用 CPW 字母：马 `H/h`，象 `E/e`。解析时也接受 UCCI / Fairy-Stockfish：马 `N/n`，象 `B/b`。
- 开局：`rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR w - - 0 1`

九宫：file 3–5，红 rank 0–2，黑 rank 7–9。河界：红 rank ≤ 4 为本方，黑 rank ≥ 5 为本方。兵过河：红 `rank >= 5`，黑 `rank <= 4`。

`Move` 是 16 位：低 7 位 from，接下来 7 位 to。没有升变（象棋兵过河只改走法，不改棋子类型）。

### 5.2 棋盘结构

`Board` 里同时有：

- `squares[90]`：格子上的棋子
- `piece_list[2][16]` + `n_pieces[2]` + `list_index[90]`：每方棋子列表，吃子时用 swap-remove
- `king_sq[2]`
- `side`、`halfmove`、`fullmove`、`hash`、`ply`
- `hist[MAX_PLY]`：每步之后的 Zobrist，用来数重复
- `undos[MAX_PLY]`：悔棋（hash、着法、被吃子、halfmove、fullmove）

`MAX_PLY = 512`。超过在 `make_move` 里抛异常；`terminal()` 在 `ply >= MAX_PLY - 1` 判和。

Zobrist：`std::mt19937_64` 固定种子 `0x9E3779B97F4A7C15`，键是棋子×格子 + 一方走子。**不含** halfmove / 重复次数，所以「同一局面」= 棋子位置 + 走方。和棋规则用的三次重复就是这个 hash。

### 5.3 走子生成

两段式：

1. `generate_pseudo_legal`：按棋子走法生成，不看是否送将。
2. `generate_legal`：对每步 `make_move`，若己方帅/将被对方攻击则丢掉，再 `unmake_move`。

各兵种（与 `attacks_from` 对齐，攻击检测用于将军/捉）：

| 棋 | 走法要点 |
| --- | --- |
| 帅/将 | 九宫内正交一格。**飞将**不是可走着，只出现在 `attacks_from`：同列无子且目标是对方将/帅 → 对方王在攻击己方王，于是任何不挡住/不离开的着都是送将，合法生成会滤掉。 |
| 仕/士 | 九宫内斜一格 |
| 相/象 | 斜两格，塞象眼空，落点必须在己方河界内 |
| 马 | 日字，蹩腿：长方向先迈一格必须空 |
| 车 | 正交滑行，可吃对方 |
| 炮 | 正交：未翻山走空，翻一座山吃对方 |
| 兵/卒 | 向前一格；过河后可横走一格。不能后退 |

`add_if_capturable`：目标空或对方子才加入。不能吃己方。

炮的 `attacks_from` **只在隔一子时为 true**（吃子），不把空滑行当攻击。将军检测因此正确：炮打将必须有炮架。

### 5.4 将军、捉

- `in_check()`：走方的帅/将被对方 `is_attacked`。
- `is_attacked(sq, by)`：遍历 `by` 方每一子，看 `attacks_from`。
- `is_chasing(us)`：实现的是**可机判子集**，不是完整亚规/世规「捉」。条件同时满足才算捉到一个子：
  1. 目标不是将/帅
  2. 目标不是**未过河**兵/卒（未过河兵不算被捉）
  3. `us` 正在攻击它
  4. 对方没有攻击它（无根；有根 = `is_attacked(sq, them)`）
  5. 攻击者里至少有一个**不是兵/卒**（仅兵去捉不算，对应「兵允许长捉」）

没做：抽将、引离、关捉、隔子捉的细分类、**长杀**、解杀还杀、等着与捉的交叉。三次重复时若循环被标成闲着，会按允许不变和棋，即使人类裁判会判长杀。

### 5.5 终局 `terminal()`

按这个顺序：

1. `ply` 触及上限 → 和，`MAX_PLY`
2. `repetition_count() >= 3` → `adjudicate_repetition()`（见下）
3. `halfmove >= 120`（吃子或走兵会清零）→ 和，`NO_PROGRESS`（六十回合）
4. 无合法着：走方负。`in_check` → `CHECKMATE`（将死），否则 `STALEMATE`（**困毙，象棋算输**，和西洋棋相反）

`repetition_count` 扫 `hist[0..=ply]` 里有多少个等于当前 hash（含现在）。开局 hash 算 1 次；同一局面第三次出现时 ≥ 3。

### 5.6 三次重复：长将 / 长捉 / 允许不变

**着法始终合法。** 循环走完之前 `terminal()` 是 ONGOING；第三次出现才裁决。这是为了 perft 和「先走完再判」一致。

`adjudicate_repetition`：

1. 从当前 ply 往回找上一次相同 hash，得到 `cycle_len`。
2. 从**当前局面往回**走 `cycle_len` 步。每一步：
   - `mover = side ^ 1`（刚走完的一方）
   - 若当前走方 `in_check()` → 这一步是**将**
   - 否则若 `is_chasing(mover)` → **捉**
   - 否则 **闲**
   - 然后 `unmake_move`，看循环里更早的一步
3. 再按保存的着法 `make_move` 走回去（局面不变，测试会 assert FEN）。
4. 判决：
   - 单方每步都是将、对方不是 → 该方负，`PERPETUAL_CHECK`
   - 双方每步都是将 → **和**
   - 否则单方每步都是将或捉、对方不是 → 该方负，`PERPETUAL_CHASE`（含一将一捉）
   - 否则和，`REPETITION`（允许不变，或双方都犯规）

测试锚点在 `tests/test_engine.py`：闲着炮来回、车长将、车追无根马、有根不算捉、仅兵攻击不算捉、困毙、将死。

### 5.7 make / unmake

`make_move` 更新格子、棋子表、王位置、Zobrist、走方、halfmove（吃或走兵清零）、黑方走完才 `fullmove++`，然后 `hist[ply] = hash`。`unmake_move` 从 `undos` 恢复。MCTS 每条模拟走下去再原路悔棋；`run()` 结束时用 FEN + ply 断言棋盘被还原，否则抛 `RuntimeError("MCTS did not restore the board")`。

Python 侧 `push` / `pop` / `push_iccs` 就是这两个。

### 5.8 pybind11

模块名 `_xiangqi`。`Board` 支持 copy、pickle（按 `sizeof(Board)` 原始字节，所以 **C++ 结构体布局变了旧 pickle 会坏**）。`perft` 和 `play_random_games` 会释放 GIL。`play_random_games` 用 Splitmix64，无共享 RNG，可并行。

编码相关绑定：`EncodeSpec`、`encode_state`（返回 numpy `(C,10,9)` float32）、`legal_indices`、`move_to_index` / `index_to_move`、`should_flip`、`flip_square`。

---

## 6. 五子棋引擎

`python/xiangqi_engine/gomoku/board.py`。自由五子，**无连珠禁手**。边长默认 15，冒烟 9。

- 黑 `STONE_BLACK = 1` 先走（`side` 初值 `BLACK`）；白 `STONE_WHITE = 2`，颜色枚举走 `RED`。
- 动作：往空交叉点放一子。ICCS 仍是 file 字母 + rank 数字，如 `h7`。
- 终局：最后一手形成横/竖/两斜任一方向 `1 + 正向连续 + 反向连续 >= 5` → 落子方胜（`FIVE`）。棋盘满 → 和（`FULL`）。没有长将、没有半回合和棋。
- `make_move` / `unmake_move` 只推一个格子下标到 `undos`。
- `in_check()` 恒为 False，好让 `terminal_value` / 评估共用象棋那套 Outcome。

这不是职业连珠。先手优势很大。训练后期平均 20 来手结束，是因为自由五子里 VCF（连续冲四）一旦会了，对局会突然变短，不是 bug。见 §16。

---

## 7. 状态编码

网络输入一律 **CHW float32**。通道数由 JSON 的 `encode.planes` × `history_length` 再加 extra 平面决定。

### 7.1 象棋平面

`n_frame_planes` = 7×our + 7×opp（默认都开）= 14。`n_extra_planes` 默认只有 `halfmove`（常数平面，值 = `halfmove / 120`）。默认 `history_length = 1` → **C = 15**，形状 `(15, 10, 9)`。

每个历史帧：走方 7 类棋各一平面、对方 7 类各一平面。格子有该子则 1，否则 0。棋子类型顺序：帅(将)、仕、相、马、车、炮、兵。

`perspective == current_player` 且黑走时 **180° 翻转**：`sq' = 89 - sq`（`flip_square`）。于是「我方」永远坐在网络的 rank 0 一侧。价值头输出的 `v` 是**当前走方**的期望结果 ∈ [-1, 1]，不是红方视角。

`side_to_move` 平面（默认关）：绝对红走为 1。只有 `perspective == absolute` 时才有用。`ones`、`fullmove` 默认关。

历史：`encode_state` 的 `history` 是**不含当前**、最老在前。窗口长度 T，不够的前面全零。Python `Encoder` 用 deque 记住最近 T-1 个已 `observe` 的棋盘。MCTS 在搜索里若 `history_length > 1`，会在下行时 `board.copy()` 推进 `search_hist`，和棋局里的 `game_past` 拼起来再编码。默认 T=1，这条路径基本不跑。

### 7.2 象棋动作索引

`index = from * 90 + to`，**在网络坐标系里**（若黑走且翻转，from/to 先 flip 再编码）。合法着大约 40–100，向量长度仍是 8100，非法槽位训练目标为 0。

推理时 `NetworkEvaluator` 对 logits **只在合法下标上 softmax**，非法概率精确为 0。训练时交叉熵**不掩码**（见 §11）：非法 logit 仍进分母。这是 AlphaZero 原论文的做法；策略头必须自己学会把质量放在合法槽。

`Encoder.play(board, index)`：index → Move（按是否翻转）→ `make_move`。MCTS 选边用的是这个 index，不是 ICCS 字符串。

### 7.3 五子棋平面

默认 our 一平面 + opp 一平面，T=1 → `(2, 15, 15)`。**不做空间翻转**（棋盘对当前走方没有「底线」）。动作就是格子下标 0..224，`place` 编码。没有「最后一手」平面，没有 8 种对称增强。这两项都是明显的增强点，还没做。

`side_to_move` 若打开：`side == 0`（白/RED）填 1。默认关。

### 7.4 Encoder 与棋盘的职责划分

- 棋局里每走一步：`board.push` 然后 `enc.observe(board)`，把**实际对局**位置记进历史窗。
- MCTS 模拟：`enc.play` 改的是搜索用的那份 board，**不** `observe`。历史靠 `past_for` + 可选的 search_hist。
- `past_for(board)`：若内部 `_current` 的 hash 和传入 board 不同，会把 `_current` 也算进过去（「你观察过的上一手」）。

---

## 8. 策略价值网络

`python/xiangqi_engine/network.py`，结构跟 AlphaZero 象棋论文同一套路：

```
Conv 3×3 + BN + ReLU          # stem，in_ch → channels
× N ResidualBlock             # 每块：Conv-BN-ReLU，Conv-BN，残差，ReLU
├─ policy: 1×1 Conv (policy_ch) + BN + ReLU → flatten → Linear → A logits
└─ value:  1×1 Conv (value_ch) + BN + ReLU → flatten → Linear hidden → ReLU → Linear 1 → tanh
```

`ResidualBlock` 第二层 BN 之后才加残差再 ReLU（pre-add BN，post-add ReLU）。没有 bias 在卷积上。

`forward` 返回 `(logits, value)`，形状 `(B, A)` 和 `(B,)`。value 已 tanh。

默认尺寸：

- 象棋 `default.json`：20 block × 256 通道，policy/value head 32 通道，value hidden 256，A=8100。10×9 上这个塔在 H200 上很小，VRAM 不是瓶颈。
- 五子棋 `gomoku.json`：6×64，A=225。这是「算法探针」尺寸，不是为了打职业。
- smoke：1×16 / 更小。

`infer()` 是单局面便捷函数：编码、前向、`masked_policy`。真正搜索走 `NetworkEvaluator.evaluate_many`。

BatchNorm 在 `net.train()`（训练）和 `net.eval()`（自我对弈 / eval / 对弈）之间行为不同。评估器里每次 forward 都 `self.net.eval()`。

---

## 9. MCTS（PUCT）

`python/xiangqi_engine/mcts.py`。一棵树一个 `MCTS` 实例。节点 `_Node`：`actions`（合法 index）、`prior` P、访问 `n`、价值累计 `w`、子节点列表、是否已扩展、终局值。

### 9.1 选择

对已扩展非终局节点：

- `Q_i = W_i / N_i`，若 `N_i = 0` 则 Q=0
- `U_i = c_puct(N_sum) * P_i * sqrt(N_sum) / (1 + N_i)`
- 选 `argmax(Q+U)`

`N_sum == 0`（根刚展开、还没模拟）时直接 `argmax(P)`。

`c_puct_base == 0`（默认）→ 常数 `c_puct`（1.5）。若 `c_puct_base > 0` 用 AlphaZero 公式：

`log((1 + N_sum + base) / base) + c_puct_init`

### 9.2 扩展与备份

叶子未扩展：若局面终局，`mark_terminal(v)`（对**走方**的价值）；若无合法着，v=-1（走方输）。否则网络给 (P, v)，`expand`。备份沿路径符号翻转：子节点返回的 v 是走方的，对父边是对手，所以 `v = -v` 再 `N+=1, W+=v`。

`terminal_value(board)`：`ONGOING→None`，`DRAW→0`，红胜则红走为 +1 否则 -1，黑胜对称。五子棋白胜走 `RED_WIN`，同一函数。

### 9.3 顺序模拟 vs 批叶子

`mcts.batch_size == 1`（或 evaluator 没有 `evaluate_many`）：循环 `_simulate`，**没有虚拟损失**。单测依赖「访问集中在先验最大边」时走这条。

`batch_size > 1`（正式配置 32）：`_simulate_batch`

1. 连续选 `n` 个叶子。下行时虚拟损失：`N += vl`，`W -= vl`（默认 vl=1）。未访问边 Q 变成 -1，同一 batch 里 PUCT 不会全挤一条边。
2. 每条选完立刻 `unmake` 把棋盘复原；叶子的编码和合法表已经拷下来。
3. 已扩展/终局的叶子直接记下 v；未扩展的按 `id(node)` 去重后 **一次** `evaluate_many`。
4. 扩展（仍未 expanded 才 expand，避免两个模拟打到同一叶子扩两次）。
5. 备份：先撤销 VL，再按真 v 做 `N+=1, W+=v`（带符号翻转）。

GPU 利用率：`batch=1` 时每模拟一次 Tiny forward，H200 功耗会掉到 ~160W，自我对弈墙钟被推理延迟绑死。`batch=32` 是目前自我对弈速度的主开关，比加大 `train.batch_size` 重要得多。

`evaluate_many`：`np.stack` 编码 → GPU → 整批 logits/value → 每行只对合法 index softmax。返回的 prior 长度 = 该叶子合法着数，不是 8100。

### 9.4 Dirichlet 噪声

`P ← (1-ε) P + ε Dirichlet(α)`，默认 ε=0.25。象棋 α=0.3，五子棋 JSON 里 0.15。

**只打在当前搜索的根上**，且 `add_noise=True`、合法着 > 1。自我对弈每次 `run(..., add_noise=True)`。评估和对弈 `add_noise=False`。

`reuse_tree=True` 时，每一步的根都是上一手的子节点，所以**每一手**都会重新涂 Dirichlet，不是只在开局根涂一次。这和「整盘棋一棵树、噪声只加在最初根」不同，探索更强，训练目标 π 也更钝。若以后要改成「只在开局加噪声」，需要在 `play_game` 里只让 ply==0 加，后面 `add_noise=False`。

### 9.5 温度与训练目标

训练要的 π **永远**是 `N/ΣN`（`_policy_from_counts`）。温度**只**用于从访问次数里采样真正走出的那一步：

- ply < `temperature_moves`（象棋 16，五子棋 JSON 12）：τ=`temperature`（1.0），`p ∝ N^{1/τ}`
- 之后 τ=0：走访问最多的边

评估和对弈始终 τ=0。不要把温度乘进 π 再当 CE 目标——代码里没有这条路径。

### 9.6 树复用

`advance(action_index)`：根换成该边的子节点，父节点压进 `_ancestors`。即使该子节点 N=0（从没被模拟到），对象仍在，只是 `expanded=False`。下一步 `run(reuse=True)` 会扩展它，相当于从空子树开始搜，但先验还在。

`retreat(plies)`：悔棋。祖先不够则 `reset()` 整棵丢掉。

`run(reuse=False)` 会新建根。自我对弈默认 reuse。评估里双方各一棵树，走出的那步**两棵都 advance**（对手的树也要降到同一局面）。对弈 UI 同一棵，悔棋 retreat。

**不要跨局共享一棵树。** 新对局 `reset` / 新 `MCTS()`。

人机里人走了一手搜索很少访问的着：`advance` 成功但子树空，下一手引擎等于冷启动。这是正常的，不是权重没加载。

### 9.7 UniformEvaluator

先验均匀、v=0。无网络时测搜索结构，也对弈 UI 权重留空时的行为。均匀 + 足够模拟仍然能走出「不送将、能吃就吃」的水平；象棋早期死亡螺旋里，这种搜索能打赢过拟合的网络。

---

## 10. 自我对弈

`selfplay.play_game`：

1. 新棋盘、encoder.reset、一棵 MCTS。
2. 每步：若已终局则停。τ 按 ply 切换。`enc.tensor(board)` 记下 **s**（走之前的平面）。`mcts.run(add_noise=True, temperature=τ, reuse=...)` 得到 π 和要走的着。
3. pending 里存 `(s, π, side)`。`push` + `observe` + `advance`。
4. 到 `max_plies` 仍未终局：按 DRAW 处理（`terminal()` 若仍 ONGOING 则 `outcome_to_red_z` 当 0）。
5. 对 pending 每一手：`z = red_z if side==RED else -red_z`。同一局里所有样本的 |z| 相同，符号按**该手走方**。黑胜时黑走的样本 z=+1，红走的 z=-1。

没有认输（resign）。没有在价值头低于阈值时截断。长局会一直走到规则终局或 `max_plies`。

多进程：`n_workers > 1` 用 `spawn`（父进程可能已经 import torch，fork+线程不安全）。每个 worker：

- `torch.set_num_threads(1)` 避免 CPU 过订阅
- 自己建 `PolicyValueNet`，`load_state_dict`（**best 的 CPU 副本**）
- 设备：`selfplay.device=auto` → `cuda:(worker_id % gpus)`
- 一盘一盘 `_play_worker(seed)`，种子 `seed + game_index`，外层再加 `cfg.seed + iteration * 10007`

`loop.run_iteration` 明确传 `state_dict=_cpu_state_dict(best)`。改回用 `net` 就会重现 §16 的螺旋。

进度：`Progress` 大约每 1/16 盘或每 30 秒打一行，带平均 plies/game 和 ETA。第一盘出来之前 spawn 可能要几分钟（32 个进程每人加载一遍 20×256）。

---

## 11. 回放池与训练

### 11.1 ReplayBuffer

`Sample`：`state` float32 CHW；`policy_index` / `policy_prob` 稀疏（只存 π>0 的槽，自我对弈里就是合法且被访问到的着）；`value` 即 z。

`sample(batch_size)`：无放回抽 min(B, len) 条，把稀疏 π 展开成 dense `(n, action_size)`。容量 `deque(maxlen=capacity)`，满了丢最旧。

`ready()`：`len >= min_size` 才训练。第一轮如果盘数太少会跳过 train（metrics["train"]=None），也就没有 eval/晋升。

### 11.2 原子写入与残缺 sidecar

`save`：写到 `path.name + ".tmp"`，再 `replace` 到目标。中断不会留下半截正式文件。

`loop.save_training_checkpoint` 先写 `.pt` 再写 `.replay.pkl`。若在写 replay 时被杀，`.pt` 可能已是新的，sidecar 仍是旧的或缺失——这没问题。若旧代码非原子 pickle dump 写到一半，`latest.replay.pkl` 会 `EOFError`。

`_try_load_replay`：先试 `iter_XXXX.replay.pkl`（与 `--checkpoint` 同 stem），再试 `data/replay/buffer.pkl`。`EOFError` / `UnpicklingError` / `OSError` → 打 warning，buffer 清空，**继续用权重续训**。不要把残缺 pickle 理解成「评估模拟次数被改了」之类；那是另一件事。

恢复用 `--resume`（`latest.pt`）或 `--checkpoint checkpoints/iter_0005.pt`。只拿到 `best.pt` 也能当初始权重，但优化器重开、回放是空的。

### 11.3 损失

```
L = w_p * CE(π, log_softmax(logits)) + w_v * MSE(v, z)
```

`CE = -(π * log_softmax(logits)).sum(dim=-1).mean()`，π 已是 dense。Adam `weight_decay` 另算（默认 1e-4）。`grad_clip` 默认 1.0。

`w_v` 默认 JSON 是 1.0。**小数据上这会让价值头先把整盘 z 背下来**（见 §16）。有效的经验值是把 `value_loss_weight` 降到 0.25 左右，并减少 `batches_per_iter`。

`epochs_per_iter` 和 `lr_schedule` 目前**没有代码读取**（恒定 lr，按 batch 数更新）。改这两个 JSON 字段不会改行为。

没有合法掩码、没有着法增强、没有在训练时对象棋做左右镜像（曾经考虑过，没做）。

---

## 12. 评估与晋升

`evaluate.play_match`：`n_games` 盘，偶数盘挑战者先走，奇数盘守卫先走。五子棋 30 盘 → 挑战者 15 黑 15 白。

每盘：`add_noise=False`，`temperature=0`，模拟次数用 `eval.mcts_simulations`（可以和自我对弈的 `mcts.simulations` 不同）。两棵树，走一步双方都 advance。

`win_rate = (wins + 0.5 * draws) / n_games`，挑战者视角。≥ `win_rate_threshold`（0.55）则 `best.load_state_dict(net)`。

自由五子、两网实力接近时，**先手几乎锁胜**：15 先 15 后 → 约 15:15，wr=0.5，不晋升。真正晋升常见形态是 **30:0**（后手也能赢）。日志里 wr 经常是 0 / 0.5 / 1.0 三档，不是评估坏了，是棋太「脆」。

`eval_every`：象棋 default 是 2，不是每轮都打。评估在训练 GPU 上**串行**下棋，盘数×模拟不要开太大，否则墙钟被评估吃掉。

---

## 13. 训练循环与断点

`run_loop`：

1. 建 `net`、`best`、Adam、空 buffer。无 resume 则 best←net。
2. 对 i = start+1 .. start+n_iters：
   - 用 best 自我对弈 → 样本进 buffer
   - buffer.ready 则 `train_batches(net, ...)`（只更新 net）
   - 到点则 eval，可能晋升
   - 一行 summary：selfplay 时分、红黑和、train loss、eval wr、是否 promoted、ETA
   - `logs/**/train.jsonl` 追加一行 JSON
   - 用 `xiangqi-plot logs/gomoku/train.jsonl` 画损失 / 评估 / 自我对弈 / 晋升（不读 torch）
   - `save_every`：写 `iter_XXXX.pt` + replay、`latest.pt` + replay、`best.pt`（只含当前 best 的 `model` 键）

checkpoint payload：

- `model`：学习者
- `best`：门控网（旧文件没有这个键时，load 会把 model 抄进 best）
- `optimizer`
- `iteration`
- `config` 快照
- `metrics` 该轮摘要

MCTS 树**不**进 checkpoint。对弈/评估每步都是从当前根搜。

`train.jsonl` 每轮一行，字段就是 `run_iteration` 返回的 dict：`iteration`、`games`、`samples`、`buffer`、`selfplay.{red,black,draw}`、`train.{loss,policy_loss,value_loss}`（buffer 不够时为 `null`）、`eval.{wins,losses,draws,win_rate}`（该轮没评估则没有这个键）、`promoted`、以及 `selfplay_sec` / `train_sec` / `eval_sec` / `sec`。resume 往同一文件追加；同一 `iteration` 出现两次时，画图脚本留最后一次。

```bash
xiangqi-plot logs/gomoku/train.jsonl
```

六张图：损失（左策略/总和，右价值；价值单独轴，否则 0.04 会被 5.2 压没）、评估 wr + W/L/D、自我对弈红/黑/和比例、平均手数 `samples/games`、回放池、墙钟。红色虚线和 ★ 是 `promoted=true`。不 import torch。

自我对弈种子 `seed + iteration * 10007`，resume 后不会和已跑轮次撞车。

---

## 14. 网页对弈

`python -m xiangqi_engine.play`：ThreadingHTTPServer，象棋默认 `127.0.0.1:8765`，五子棋 `8766`。静态文件在 `play/static/`。API：

| 路径 | 作用 |
| --- | --- |
| `GET /api/state` | 棋盘、合法着、FEN、权重路径 |
| `POST /api/new` | 新局；body 含 red/black/simulations/checkpoint |
| `POST /api/move` | 人走 ICCS |
| `POST /api/undo` | `plies` 或 `human_turn`（人机连悔引擎应着） |
| `POST /api/ai` | 引擎走一步 |

引擎：**CPU** `NetworkEvaluator`，无 Dirichlet，τ=0，模拟次数用表单/`play.simulations`（常比训练 800/400 小，例如 200）。和训练不是同一分布，人机体感会偏弱或偏「死」，见 §16.5。

权重框必须显示**当前会话真正加载的路径**（`state.checkpoint`）。`resolved_checkpoint` 对存在的文件收成绝对路径。空字符串 → UniformEvaluator，不需要已有网络。启动时若只读 JSON 里的 `play.checkpoint` 而忽略 `--checkpoint`，UI 会撒谎；已经修过：`server.main` 把 CLI 传进 `new_game`。

`PlaySession._bind_search`：若任一方是 AI，开局就先 `run` 一次（reuse），避免第一手完全冷树。人走后再 `advance`。

界面：象棋点从–到，五子棋点交叉点；棋盘 CSS 随窗口缩放。`play` 段改端口/默认角色不影响训练。

---

## 15. 测试在锁什么

`pytest` 路径 `tests/`。和实现强绑定的几条：

- `test_engine.py`：开局 44 着、UCCI 字母、CPW perft、困毙、将死、三次重复和、长将负、长捉负、有根/兵捉否定、copy/pickle
- `test_encode.py`：平面形状、黑走翻转、动作 index 往返
- `test_mcts.py`：PUCT 选边、batch 路径、树 advance/retreat、终局价值符号
- `test_selfplay.py` / `test_train.py` / `test_loop.py`：样本 z 的符号、门控用 best、replay 坏文件跳过
- `test_play.py`：checkpoint 路径出现在 state 里
- `test_gomoku.py`：五连、满盘、工厂切换
- `test_network.py`：输出形状、masked softmax

改规则或损失时先跑这一套。长将的测试会 assert 裁决后 FEN 不变（unmake/make 必须成对）。

---

## 16. 训练经验（真机日志）

下面是实际训练反馈，不是「应该会怎样」。配置以当时改过的为准，仓库里的 JSON 默认值可能仍是旧的。

### 16.1 象棋第一轮：价值过拟合 + 用学习者造棋 → 死亡螺旋

设定：`default.json` 量级，20×256，800 sims，`value_loss_weight=1`，`batches_per_iter=1000`，自我对弈用**当时的最新 net**（还没有 best 门控）。

大约 7 轮、自我对弈每轮约 10 小时量级：

- 策略损失 3.77 → 2.27：先学会合法槽，再开始往大约 10 步候选上集中。这部分看起来「在学」。
- 价值损失 0.47 → **一轮里掉到 0.04**：不是棋力飞跃，是价值头把「这一盘的 z」背下来。同一局所有样本共享一个 z，1000 个大 batch 足够记住棋谱结局。
- 评估对象是从未晋升过的**初始网 + MCTS**。先 10–10，然后 **0–20、0–20**。学习者比「均匀先验 + 搜索」还弱。

机制：过拟合的 v 污染 MCTS 的 backup → 搜索信错的价值 → 自我对弈样本更差 → 再训练更偏。因为造棋用的就是这份坏净，没有冻结的参照系，出不来。

对策（已写进 `loop.py`）：自我对弈始终 `best`；只有 eval 过线才替换。这是 AGZ 2017 的门，不是 AZ 2018。**小算力必须上门控。** 2018 论文能用最新网，是因为每步模拟极多、自我对弈量极大，噪声平均掉了；这里每轮几百盘做不到。

### 16.2 五子棋同样的价值塌缩（改超参之前）

6×64，200 sims，400 batches，`value_weight=1`，大约 11 轮：

- 策略损失卡在 ~5.24（`log(225) ≈ 5.42`，几乎还是均匀）
- 价值 ~0.05 又是背结局
- eval 0:16，偶发一次噪声 8:8

结论：不是五子棋学不会，是更新太猛、v 权太重、模拟太少，π 目标本身就很平。

### 16.3 改完以后五子棋真的在变强

你这边后来有效的旋钮（相对 `gomoku.json` 仓库默认）：

| 项 | 仓库默认 | 有效方向 |
| --- | --- | --- |
| `batches_per_iter` | 400 | **80**（少刷几遍同一批棋） |
| `value_loss_weight` | 1.0 | **0.25** |
| `mcts.simulations` | 200 | **400**（π 更尖，比加 batch 更有用） |
| `eval.n_games` / `mcts_simulations` | 16 / 100 | **30 / 200** |
| 规模 | 128 盘 / 32 worker | 曾用 64 worker，再后来 512 盘/轮、buffer 30 万 |

结果：

- 第 4 轮 **30:0 晋升**（对初始 best）。自由五子里这表示后手也能赢，不是刷先手。
- 约 50 轮策略 5.36 → 4.20；对局 55 手 → 23 手；7 次晋升。eval 仍大量落在 0 / 0.5 / 1.0。
- 阶段 1 的棋：几乎只会把一条线往前接，还不会两边开花、也谈不上 VCF。这是正常课程——先学「五连这件东西存在」。
- ~100 轮策略 ~4.0。这个数接近 **400 次模拟 + 每步 Dirichlet** 下访问分布的熵，再堆 `batches_per_iter` 也压不下去。要更尖的 π，加 sims 比加梯度步更有效。
- 第 102 轮又晋升。
- 第 363 轮附近：策略 3.52，约 21 手/局，自我对弈黑胜 483 vs 白胜 29。人已经**后手和先手都很难赢这份网**。短对局是实力上升后的正确现象（会杀了），不要当成坍缩。

6×64、几十轮解决不了「随便一个会五子的人」。对「年对局 < 500」的业余，一夜训出来的这份网已经在那个区间里。它的任务是证明闭环，不是打职业连珠。

### 16.4 速度：洞在自我对弈，不在 train

H200 上 20×256、batch 4096 的梯度更新几分钟；自我对弈按 800 sims × 每步一次（或很少）GPU forward 要数小时。`mcts.batch_size=1` 时卡吃不满。`batch_size=32` + `virtual_loss=1` 是补这个洞的。Worker 数、`n_games_per_iter`、sims 才是墙钟旋钮。训练 GPU 和自我对弈抢卡时用 `CUDA_VISIBLE_DEVICES` 和 `selfplay.gpus`。

### 16.5 人机和训练不是同一套搜索

| | 自我对弈 | 网页对弈 |
| --- | --- | --- |
| 网络 | GPU 上的 best | **CPU** |
| Dirichlet | 每步都有 | 无 |
| 温度 | 前 N 手 τ=1 | 永远 0 |
| 模拟 | 训练配置（200–800） | 表单，常常更少 |
| 树 | 每步都从上一手子树续 | 人走冷门着则子树空 |

所以「jsonl 里已经 30:0 了，我怎么还是能赢」往往是：UI 模拟少、CPU 慢所以你不敢开高、没有噪声时引擎更死、或权重框其实没加载到你以为的那个 `iter_XXXX.pt`（修过：界面展示会话里的绝对路径）。

### 16.6 别把基础设施事故当成算法

- 写到一半的 `latest.replay.pkl` → `--resume` 炸。现在原子写 + 跳过坏 sidecar。应从最近完好的 `iter_XXXX.pt` 续。
- 权重 UI 曾经显示配置默认路径而不是 `--checkpoint`。
- 象棋 `rules` JSON 改了引擎也不会跟。

### 16.7 目标感

象棋目标是赢家里人，不是世界冠军。五子棋是课程验证。评估 30 盘打出 15:15 时不要加盘数硬刷 0.55——先手锁死下那是平手；要晋升得看到后手也开始赢（30:0 或至少明显 >0.55 且不是纯先手）。

---

## 17. 刻意没做的、以及已知简化

| 项 | 状态 |
| --- | --- |
| 认输 | 无 |
| 五子棋最后一手平面 | 无 |
| 五子棋 8 对称增强 | 无 |
| 象棋训练时左右镜像 | 无（你曾经明确先不做） |
| 长杀 / 完整亚规捉 | 无；三次重复可能被判允许不变 |
| 训练 CE 合法掩码 | 无（推理才掩） |
| `epochs_per_iter` / `lr_schedule` JSON | 未接线 |
| `rules.*` JSON | 未接线，C++ 写死 |
| AlphaZero 2018「永远最新网」 | 已废弃；小算力用 AGZ 门控 |
| 把 MCTS 树写入 checkpoint | 无 |
| 连珠禁手 / 职业五子规则 | 无 |
| 网络在 `__init__.py` 里 import | 禁止，以免拖累纯引擎用途 |

这些不是疏忽清单里的「下一步必做」，而是读代码时不要脑补已经存在的部分。

---

## 18. 建议的阅读顺序（对着文件）

1. `config/default.json` + `config/gomoku.json` — 所有数字的家
2. `game.py` — 两套棋如何共用
3. `include/xiangqi/board.hpp` 扫类型，再 `src/board.cpp` 的 `generate_legal` / `terminal` / `adjudicate_repetition` / `is_chasing`
4. `include/xiangqi/encode.hpp` + `encode.cpp` — 平面和 8100 槽
5. `mcts.py` — 先 `_simulate`，再 `_simulate_batch`
6. `selfplay.py` 的 z 符号 + `loop.py` 的 `run_iteration`（盯着 `best` 不是 `net`）
7. `train.py` 的无掩码 CE
8. `evaluate.py` 的颜色轮换和 win_rate
9. `play/session.py` — 和训练分布的差异
10. `tests/test_engine.py` 长将/长捉用例 — 规则的可执行注释

改超参时看 §16 再动 `value_loss_weight`、`batches_per_iter`、`simulations`、是否门控。这四个比改网络层数更容易把实验救活或弄死。
