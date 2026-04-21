# EditKG 迁移与显存/LLM 优化记录

## 1. 本次迁移的目标

本次修改围绕以下目标展开：

- 适配新服务器上的数据集与模型路径
- 保证 `amazon-book` 数据能被主训练流程和 LLM 流程正确读取
- 在 48GB 显存服务器上尽量保留带 LLM 的训练逻辑
- 在不明显影响最终训练效果的前提下，控制总体训练时长

当前新的关键路径如下：

- 数据集目录：`/root/autodl-tmp/projects/EditKG/data/datasets/amazon-book`
- LoRA 权重目录：`/root/autodl-tmp/projects/EditKG/data/models/checkpoint-37736`
- Qwen3-4B-Instruct-2507 本地目录：
  `/root/autodl-tmp/data/models/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507`

## 2. 路径与工程兼容性修改

### 2.1 路径解析

新增了统一路径解析逻辑，避免代码里硬编码旧服务器路径。核心是：

- 统一从项目根与 `data/` 目录推导数据集路径
- 默认数据集根目录改为新的服务器路径
- 默认 LoRA 路径改为新的 checkpoint 路径

相关文件：

- [utils/path_utils.py](/root/autodl-tmp/projects/EditKG/EditKG/utils/path_utils.py)
- [utils/parser.py](/root/autodl-tmp/projects/EditKG/EditKG/utils/parser.py)
- [utils/data_loader.py](/root/autodl-tmp/projects/EditKG/EditKG/utils/data_loader.py)
- [main.py](/root/autodl-tmp/projects/EditKG/EditKG/main.py)

### 2.2 缓存文件落盘位置修正

原代码中部分缓存文件会直接写到当前工作目录，迁移后容易写错位置。现在这些缓存跟随数据集目录保存，避免多服务器、多工作目录下混乱。

主要涉及：

- `item_rel_mask_rev.pkl`
- `item_pair_pmi.pkl`

## 3. 已修复的代码问题

### 3.1 `main.py` 延迟加载 LLM

原逻辑中即使不需要 LLM，训练启动时也可能触发相关依赖加载。现在改为只在需要时才初始化 `LLMScorer`。

作用：

- 减少不必要的启动失败风险
- 避免无 LLM 实验也被 `transformers/peft` 依赖拖住

### 3.2 `generate_hard_negatives.py` 接口失配

旧代码里 `generate_hard_negatives.py` 与 `llm_data_utils.py` 的接口已经不一致，直接运行会报错。现在已经修复，保证相关工具脚本可用。

## 4. 显存分析结论

### 4.1 主训练模型的显存情况

对主推荐模型做过单步实测：

- `batch_size=4096` 时，主训练图本身反向后保留显存接近 40GB
- 多模态特征不是主要瓶颈，关闭后节省并不明显

因此在 48GB 卡上：

- 主模型单独训练可以尝试跑
- 但如果再把 Qwen 4B 教师模型长期常驻在同一进程和同一张卡里，风险很高

### 4.2 为什么把训练 `batch_size` 改成 2048

当前训练里有一类非常吃显存的张量，其规模直接和 `batch_size` 成正比。例如用户 embedding 与全物品 embedding 做乘法时，会产生形如：

- `[batch_size, n_items]`

的打分矩阵。

所以把训练参数从：

- `batch_size=4096`

改成：

- `batch_size=2048`

是本次 48GB 服务器适配中最关键的一步之一。

这项改动的思路是：

- 优先保住主训练过程的稳定性
- 不动模型结构和损失设计
- 用更小 batch 直接降低训练激活显存

## 5. LLM 机制改造思路

### 5.1 原始逻辑的问题

原始带 LLM 逻辑虽然是“每 3 轮更新一次候选 KG 并打分”，但 LLM 本体会在训练进程内长期驻留。对于 48GB 显存服务器，这是主要风险源。

### 5.2 现在的改造方向

现在改为：

- 候选 KG 仍然按原算法逻辑周期性刷新
- 刷新后把候选三元组保存到磁盘
- 启动独立子进程执行 LLM 打分
- 打分结束后把结果保存到磁盘并退出
- 主训练进程读取分数，再继续训练

这样做的目的：

- 保持“候选 KG 会变，LLM 分数也重新计算”的原始逻辑
- 避免 Qwen 4B 长期常驻训练进程

相关文件：

- [main.py](/root/autodl-tmp/projects/EditKG/EditKG/main.py)
- [utils/precompute_llm_scores.py](/root/autodl-tmp/projects/EditKG/EditKG/utils/precompute_llm_scores.py)

### 5.3 三元组历史缓存

新增了三元组级缓存：

- key: `(head, relation, tail)`
- value: LLM 分数

目的：

- 如果后续 epoch 生成的新候选 KG 与历史候选有重叠，则可以直接复用旧分数
- 只对新增三元组继续调用 LLM

在 `amazon-book` 当前第一次候选 KG 刷新中，首轮候选共有 `820604` 条，且首轮内部没有重复，所以首轮缓存命中为 0；但后续 epoch 的候选 KG 有可能与之前重叠，因此缓存仍然有价值。

## 6. LLM 打分语义调整

### 6.1 原始实现

原始 `LLMScorer` 的 prompt 要求模型只输出：

- `0`
- `1`

然后代码把生成文本解析为：

- `0.0`
- `1.0`

如果解析失败，则退回 `0.5`

也就是说，原始实现本质是“离散输出 + 失败兜底”。

### 6.2 现在采用的连续分数方案

现在采用的方案是：

- 保持同一条 prompt 的判别语义
- 不再通过 `generate()` 解码文本
- 直接读取模型在当前位置输出 token `0` 与 token `1` 的 logits
- 对二者做 softmax
- 使用 `P(token=1)` 作为最终连续分数

这样得到的值天然位于 `[0,1]` 之间，例如：

- `0.17`
- `0.63`
- `0.94`

它的优点是：

- 分数连续，适合蒸馏
- 语义与“判断真/假倾向”一致
- 比文本生成更直接

对应文件：

- [modules/llm_scorer.py](/root/autodl-tmp/projects/EditKG/EditKG/modules/llm_scorer.py)

## 7. 为了稳定运行做的 LLM 侧工程增强

### 7.1 自动降批重试

由于 LLM 打分对显存很敏感，已经在预计算脚本中加入自动降批逻辑：

- 先按设定的 `llm_batch_size` 尝试
- 如果 OOM，则自动减半重试
- 直到找到能跑通的 batch size

这项改动的目的：

- 不需要人工不断试参数
- 在不同服务器、不同显存碎片状态下自适应

### 7.2 CUDA allocator 设置

在 LLM 预计算脚本中加入了：

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

目的是减少分配碎片导致的失败概率。

## 8. 显存监测插桩

为了准确定位“到底是哪里耗显存”，本次新增了统一显存监测工具：

- [utils/memory_monitor.py](/root/autodl-tmp/projects/EditKG/EditKG/utils/memory_monitor.py)

并在以下关键阶段插入监测点：

- `main.py` 启动后
- 数据加载完成后
- NPMI 计算完成后
- 模型初始化后
- 优化器初始化后
- 每 3 轮候选 KG 刷新前后
- LLM 子进程打分前后
- 每轮训练中首个 batch，以及按间隔采样的 batch：
  - forward 前
  - forward 后
  - backward 后
  - optimizer.step 后
- `model.generate(for_kgc=True)` 前后
- LLM 子进程内部：
  - 模型加载完成后
  - tokenizer 完成后
  - LLM 前向完成后

新增参数：

- `--mem_debug`
- `--mem_debug_interval`

使用方式示例：

```bash
python main.py \
  --dataset amazon-book \
  --gpu_id 0 \
  --mem_debug \
  --mem_debug_interval 50
```

日志内容会打印为：

- 当前 allocated 显存
- reserved 显存
- max allocated 峰值
- max reserved 峰值

这样可以区分：

- 是模型常驻显存高
- 还是某一段 forward/backward 瞬时拉高
- 还是 LLM 打分阶段独占了主要显存

## 9. 当前与训练时长相关的判断

用户的目标是：

- 之前在 H20 服务器上，`batch_size=4096` 全训练约 7 小时
- 现在希望在当前服务器上总时长尽量不超过 16 小时

当前判断如下：

- 主训练 `batch_size` 从 4096 降到 2048 后，训练 step 数会增加
- 同时带 LLM 时，首轮候选 KG 的打分是额外时间开销
- 如果完全沿用“文本生成式打分”，时间会明显超预算
- 因此才把 LLM 改成了“连续概率分数 + 子进程 + 自动降批”的方案，以尽量把总时长压到可接受区间

最终是否能控制在 16 小时以内，取决于：

- 当前机器上主训练每 epoch 的真实耗时
- LLM 在稳定 batch size 下对每次候选 KG 刷新的耗时
- 后续 epoch 间候选 KG 与缓存的重叠率

换句话说：

- 16 小时是本次优化的目标
- 但不会为了机械压时长而牺牲最终训练逻辑和模型效果

## 10. 当前正式测试使用参数

当前正式测试参数为：

```bash
python main.py \
  --dataset amazon-book \
  --lr 0.0005 \
  --dim 128 \
  --channel 128 \
  --context_hops 2 \
  --margin 0.2 \
  --max_iter 2 \
  --batch_size 2048 \
  --test_batch_size 2048 \
  --gpu_id 0 \
  --seed 2023 \
  --llm_model_path /root/autodl-tmp/data/models/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507 \
  --llm_adapter_path /root/autodl-tmp/projects/EditKG/data/models/checkpoint-37736 \
  --llm_score_mode subprocess
```

说明：

- `batch_size` 已从更高配置降到 `2048`
- LLM 采用子进程打分，不再常驻训练进程
- LLM 分数采用连续概率形式

## 11. 后续会继续更新的内容

当前已确认的运行时结论：

- `llm_batch_size=128` 在当前 48GB 服务器上仍会 OOM
- `llm_batch_size=96` 可以稳定跑通
- `llm_batch_size=64` 也可以稳定跑通
- 在 4096 条候选三元组的小样本压测中：
  - `batch_size=64` 可以完整打分完成
  - `batch_size=96` 也可以完整打分完成
  - 两者吞吐接近，因此正式训练当前采用 `llm_batch_size=96`

按 4096 条样本的真实压测结果粗略估计，首轮 `820604` 条候选三元组的 LLM 打分耗时为数小时级，但仍明显优于原先的文本生成式打分方案。

本文件后续还会继续补充：

- 正式训练的首轮 LLM 打分实际耗时
- 完整测试是否跑通
- 最终总耗时与关键日志结论
