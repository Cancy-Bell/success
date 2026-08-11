# AURC 端到端图模型实验进展总结

更新时间：2026-08-10

本文档用于总结近期模型实现修改、每次修改的动机、修改后的实验结果，以及当前下一步实验重点。当前主目标是提高 Official Token macro-F1，同时让 HetGAT 的高层立场推理反馈真正帮助 Final BIO，而不是破坏 Initial BIO 已经学到的边界信息。

## 1. 当前模型整体流程

当前端到端框架如下：

```text
Topic + Sentence -> Joint BERT -> sentence WordPiece 语义表示
Topic            -> Topic BERT -> Topic CLS
spaCy dependency -> WordPiece 邻接矩阵 A' = M A_spacy M^T

WordPiece 语义表示 + dependency adjacency
-> Dependency GCN
-> Semantic-Syntax Fusion
-> Initial BIO emissions E0
-> Initial BIO CRF

训练时 Gold BIO / 推理时 Initial predicted BIO
-> AU extraction
-> Topic-guided AU attention pooling
-> Document attention pooling
-> Batch-level AU/Document graph
-> HetGAT
-> AU stance + Document stance
-> Token feedback correction
-> Final BIO emissions
-> Final BIO CRF
```

核心公式：

```text
A' = M A_spacy M^T

Z_ss = Fusion(H_BERT, H_syn)
H = Dropout(GELU(Z_ss))

E0 = W_BIO H + b_BIO
Y_initial = CRF(E0)

h_AU_k = sum_i alpha_i^k V_i
alpha_i^k = softmax((W_Q h_T)^T (W_K h_i) / sqrt(d))
其中 attention 只在 AU_k 内部 token 上计算

L = L_BIO + L_AU + L_Document
L_BIO = L_FinalCRF + 0.3 L_InitialCRF + 0.5 L_OfficialTokenCE
```

加入 feedback gate 约束之前：

```text
E_final = E0 + g * R
g = sigmoid(W_g [h_i; c_i^AU; c_i^D] + b_g)
```

加入 feedback gate 约束之后：

```text
p0_i = softmax(E0_i)
u_i = -sum_c p0_i,c log(p0_i,c) / log(5)

s_i = sigmoid((u_i - tau) / T)
g_i = gate_max * s_i * sigmoid(W_g [h_i; c_i^AU; c_i^D] + b_g)

E_final_i = E0_i + g_i * R_i
```

含义：

- 如果 Initial BIO 很确定，归一化熵 `u_i` 较低，则 feedback 被压小。
- 如果 Initial BIO 不确定，`u_i` 较高，则允许 HetGAT / stance feedback 对 token emission 进行修正。

## 2. 实验结果总表

| 实验 | Best Epoch | Dev Official Token F1 | Test Official Token F1 | Test Segment F1 | Test Sentence F1 | Test AU Span F1 | Test AU Stance F1 | Test Entity F1 | Test Doc F1 | Test Initial BIO F1 | Test Final BIO F1 | Final-Initial |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-08-07 batch32 early full model | 19 | 0.7123 | 0.6988 | 0.6608 | 0.7181 | 0.4630 | 0.8711 | 0.4034 | 0.7113 | NA | NA | NA |
| 2026-08-09 old best baseline | 21 | 0.7032 | 0.7040 | 0.6614 | 0.7228 | 0.4629 | 0.8787 | 0.4151 | 0.7159 | 0.6186 | 0.6257 | +0.0071 |
| 2026-08-10 O-bias 0.25 | 10 | 0.6978 | 0.6897 | 0.6532 | 0.7100 | 0.4599 | 0.8656 | 0.4046 | 0.7162 | 0.6206 | 0.6107 | -0.0098 |
| 2026-08-10 O-bias 0.325 | 3 | 0.6330 | 0.6083 | 0.5942 | 0.6374 | 0.2463 | 0.7226 | 0.1965 | 0.5883 | 0.4716 | 0.4704 | -0.0012 |
| 2026-08-10 O-bias 0.4 | 10 | 0.6857 | 0.6827 | 0.6421 | 0.7028 | 0.4461 | 0.8645 | 0.3985 | 0.7089 | 0.6179 | 0.6049 | -0.0130 |
| 2026-08-10 conservative feedback gate | 15 | 0.7003 | 0.6939 | 0.6541 | 0.7208 | 0.4632 | 0.8716 | 0.4087 | 0.7137 | 0.6118 | 0.6141 | +0.0023 |

当前 Test 上最好的结果仍然是：

```text
2026-08-09 old best baseline
Test Official Token F1 = 0.7040
Test Entity F1 = 0.4151
Test Final BIO - Initial BIO = +0.0071
```

## 3. 实验时间线

### 3.1 O-bias 和 Gate 约束之前的基线实验

路径：

```text
F:\AURC-master\models\batch_size_comparison\run_20260809_081411\batch_size_32
```

主要设置：

```text
batch_size = 32
该实验中 initial_o_bias 约为 0.325
完整端到端图模型
best checkpoint 按 Dev Official Token macro-F1 选择
```

最好结果：

```text
Best epoch = 21
Dev Official Token F1 = 0.7032
Test Official Token F1 = 0.7040
Test Segment F1 = 0.6614
Test Sentence F1 = 0.7228
Test AU Span F1 = 0.4629
Test AU Stance F1 = 0.8787
Test Entity F1 = 0.4151
Test Document F1 = 0.7159
Initial BIO F1 = 0.6186
Final BIO F1 = 0.6257
Feedback delta = +0.0071
```

解释：

这是目前观察到的最强 Test 结果。最重要的积极信号是 Final BIO 明显高于 Initial BIO，说明图推理和 stance feedback 在这一轮中确实带来了 token-level 修正收益。

### 3.2 精简 official-style prediction JSON 输出

修改动机：

最开始生成的 official-style JSON 中包含太多诊断字段，例如 gate、概率、attention 权重、AU 细节、alignment debug 等。人工比较 Initial CRF 和 Final BIO 差异时不够直观。

修改文件：

```text
src/prediction_io.py
convert_predictions_to_aurc.py
tests/test_prediction_io.py
```

新的精简输出字段：

```text
官方原始字段：
Cross-Domain
In-Domain
is_argument
sentence
sentence_hash
sentence_level_stance
tokenized_sentence_bert
tokenized_sentence_bert_labels
tokenized_sentence_bert_single_labels
tokenized_sentence_spacy
tokenized_sentence_spacy_labels

新增对照字段：
model_sentence_wordpieces
gold_bio
initial_crf_bio
final_bio
official_gold_labels
official_initial_labels
official_pred_labels
```

已生成文件：

```text
F:\AURC-master\models\batch_size_comparison\run_20260809_081411\batch_size_32\predictions\test_best_aurc.json
```

验证结果：

```text
topics = 8
rows = 1200
has_feedback_gate = False
has_probs = False
gold_bio / initial_crf_bio / final_bio 长度均与 model WordPieces 对齐
```

影响：

这只是输出格式修改，不改变模型训练。

### 3.3 第一次 O-emission Bias Sweep 尝试

路径：

```text
F:\AURC-master\models\o_bias_sweep\run_20260809_220219
```

发现的问题：

summary 中记录了三个 O-bias 值，但实际 `console.log` 中的训练命令没有包含：

```text
--initial_o_bias
```

因此这轮不是真正有效的 O-bias sweep，本质上更像是重复跑了几次旧默认配置。

采取的修复：

在以下文件中加入命令检查：

```text
run_o_bias_sweep.py
tests/test_o_bias_sweep.py
```

新行为：

如果 sweep 生成的训练命令中不包含 `--initial_o_bias`，脚本会直接报错停止，避免白跑一整晚。

### 3.4 真正生效的 O-emission Bias Sweep

路径：

```text
F:\AURC-master\models\o_bias_sweep\run_20260810_081140
```

测试取值：

```text
initial_o_bias = 0.25
initial_o_bias = 0.325
initial_o_bias = 0.4
```

这轮是有效的，因为日志中出现了：

```text
Final O-emission bias: learnable scalar initialized to 0.250
Final O-emission bias: learnable scalar initialized to 0.325
Final O-emission bias: learnable scalar initialized to 0.400
```

结果：

```text
O-bias 0.25:
Best epoch = 10
Dev Official Token F1 = 0.6978
Test Official Token F1 = 0.6897
Initial BIO F1 = 0.6206
Final BIO F1 = 0.6107
Delta = -0.0098

O-bias 0.325:
Best epoch = 3
Dev Official Token F1 = 0.6330
Test Official Token F1 = 0.6083
Initial BIO F1 = 0.4716
Final BIO F1 = 0.4704
Delta = -0.0012

O-bias 0.4:
Best epoch = 10
Dev Official Token F1 = 0.6857
Test Official Token F1 = 0.6827
Initial BIO F1 = 0.6179
Final BIO F1 = 0.6049
Delta = -0.0130
```

结论：

O-bias 训练没有提升模型表现，反而经常让 Final BIO 低于 Initial BIO。说明直接提高 O emission 的偏置会让模型过于保守，容易把本来应该是 Pro/Con 的 argument token 推向 `O`。

决策：

不再把 O-bias sweep 作为主要提升方向。默认 `initial_o_bias` 重置为：

```text
initial_o_bias = 0.0
```

### 3.5 保守版 Feedback Gate 约束

修改动机：

O-bias 没有解决核心问题。更直接的问题是 HetGAT / stance feedback 对 token emission 的修改过于自由。理想行为应该是：

```text
如果 Initial BIO 很确定：feedback 应该尽量少动。
如果 Initial BIO 不确定：feedback 可以修正 token polarity / boundary。
```

修改文件：

```text
src/models.py
src/run_AURC_token.py
run_batch_size_experiments.py
run_o_bias_sweep.py
tests/test_model_smoke.py
tests/test_experiment_command.py
```

保守版设置：

```text
initial_o_bias = 0.0
feedback_gate_max = 0.5
feedback_uncertainty_threshold = 0.6
feedback_uncertainty_temperature = 0.1
feedback_gate_bias_init = -1.0
```

路径：

```text
F:\AURC-master\models\batch_size_comparison\run_20260810_152547\batch_size_32
```

结果：

```text
Best epoch = 15
Dev Official Token F1 = 0.7003
Test Official Token F1 = 0.6939
Test Segment F1 = 0.6541
Test Sentence F1 = 0.7208
Test AU Span F1 = 0.4632
Test AU Stance F1 = 0.8716
Test Entity F1 = 0.4087
Test Document F1 = 0.7137
Initial BIO F1 = 0.6118
Final BIO F1 = 0.6141
Delta = +0.0023
```

解释：

这一版修复了 O-bias 最明显的问题：Final BIO 又重新高于 Initial BIO。但是 gate 过于保守，Final BIO 只提升 `+0.0023`，明显低于旧最好结果中的 `+0.0071`，Test Official Token F1 也没有超过旧最好结果。

决策：

保留“不确定性门控”的思想，但需要适当放开 gate。

### 3.6 中等放开版 Feedback Gate：当前代码，尚未完成实验

修改动机：

保守版 gate 能减少破坏，但 HetGAT feedback 使用不足。下一步是在仍然保护高置信 Initial BIO 的前提下，允许更多合理修正。

当前默认设置：

```text
initial_o_bias = 0.0
feedback_gate_max = 0.75
feedback_uncertainty_threshold = 0.5
feedback_uncertainty_temperature = 0.1
feedback_gate_bias_init = -0.5
```

修改文件：

```text
src/models.py
src/run_AURC_token.py
run_batch_size_experiments.py
run_o_bias_sweep.py
tests/test_experiment_command.py
```

验证：

```text
D:\anaconda\envs\aurc\python.exe -m unittest discover -s tests -p "test_*.py" -v

Ran 30 tests
OK
```

训练开始时应看到的日志：

```text
Feedback gate constraint: max=0.750 uncertainty_threshold=0.500 temperature=0.100 bias_init=-0.500
```

状态：

该配置已经实现并通过测试，但当前日志中还没有看到这组配置跑完后的完整实验结果。

推荐运行命令：

```powershell
D:\anaconda\envs\aurc\python.exe F:\AURC-master\run_batch_size_experiments.py --batch_sizes 32
```

## 4. 当前总体判断

目前观察到的最佳 Test 结果仍然是 2026-08-09 baseline：

```text
Test Official Token F1 = 0.7040
Final BIO improvement = +0.0071
```

真正生效的 O-bias sweep 说明，直接调高 `O` emission 并不是正确方向：

```text
O-bias 0.25: Final BIO delta = -0.0098
O-bias 0.325: Final BIO delta = -0.0012
O-bias 0.4: Final BIO delta = -0.0130
```

保守版 uncertainty gate 比 O-bias 方向更合理：

```text
Final BIO delta = +0.0023
```

但是它太保守。当前代码已经改为一个更合理的中间点：

```text
gate_max = 0.75
threshold = 0.5
bias_init = -0.5
```

下一轮实验要回答的问题是：

```text
中等放开版 gate 是否能保持 Final BIO > Initial BIO，
同时让 Test Official Token F1 回到或超过 0.7040？
```

## 5. 下一轮实验对比重点

下一次 batch-size 32 实验完成后，优先和旧最好结果比较：

| 指标 | 需要超过的目标 |
|---|---:|
| Dev Official Token F1 | 0.7032 |
| Test Official Token F1 | 0.7040 |
| Test Segment F1 | 0.6614 |
| Test Sentence F1 | 0.7228 |
| Test AU Span F1 | 0.4629 |
| Test AU Stance F1 | 0.8787 |
| Test Entity F1 | 0.4151 |
| Test Final BIO - Initial BIO | +0.0071 |

最重要的成功条件：

```text
Test Official Token F1 >= 0.7040
并且
Final BIO F1 > Initial BIO F1
```

最理想的信号：

```text
Final BIO delta 接近或超过 +0.0071
```

危险信号：

```text
Final BIO delta 再次变成负数
```

如果中等放开版 gate 仍然不理想，下一步应该控制变量 sweep feedback 参数，而不是继续 sweep O-bias：

```text
feedback_gate_max: 0.6, 0.75, 0.9
feedback_uncertainty_threshold: 0.45, 0.5, 0.55
feedback_gate_bias_init: -0.75, -0.5, -0.25
```

但不要一开始就跑全部组合，除非运行时间可以接受。建议先跑当前中等放开版 batch-size 32 单次实验。
