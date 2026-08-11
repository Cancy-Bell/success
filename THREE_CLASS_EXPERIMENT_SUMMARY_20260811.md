# 三分类端到端图模型实验结果总结（2026-08-11）

本文总结实验：

```text
run_20260811_174549 / batch_size_32
```

结果目录：

```text
F:\AURC-master\models\batch_size_comparison\run_20260811_174549\batch_size_32
```

本次实验完成了 batch size 32；配置文件虽然声明还要运行 batch size 64，但当前结果目录中没有 `batch_size_64`，因此本文只分析已经完成的 batch size 32。

---

## 一、本次实验相对旧模型的修改

### 1. Token 标签由五分类改为三分类

旧标签空间：

\[
\{O,B\text{-}Pro,I\text{-}Pro,B\text{-}Con,I\text{-}Con\}
\]

新标签空间：

\[
\boxed{\{non,con,pro\}}
\]

Initial classifier、Initial CRF、token feedback、Final emission 和 Final CRF 均直接工作在三分类空间中，不再进行五分类到三分类的折叠。

由于删除了 B/I 标签，AU 只能根据连续且立场相同的 token 恢复。例如：

```text
non pro pro pro non con con
    └─ AU-Pro ─┘   └AU-Con┘
```

相邻且同立场、两者之间没有 `non` 的两个 AU，在三分类表示下无法区分，会被合并成一个连续 AU。

### 2. HetGAT 删除关系缩放参数

旧消息聚合：

\[
m_j=
\sum_r\sum_{i\in\mathcal N_r(j)}
\alpha_{ij}^{r}\gamma_r z_i^r
\]

新消息聚合：

\[
\boxed{
m_j=
\sum_r\sum_{i\in\mathcal N_r(j)}
\alpha_{ij}^{r}z_i^r
}
\]

其中关系权重完全由注意力系数 \(\alpha_{ij}^{r}\) 决定，不再额外乘可学习的 \(\gamma_r\)。

### 3. AU 图立场直接使用概率

AU stance classifier 仍进行 Pro/Con 二分类：

\[
P_{AU}=[P(Pro),P(Con)]
\]

映射到官方三分类顺序 `non/con/pro`：

\[
\boxed{
G_i=[0,P(Con),P(Pro)]
}
\]

因为：

\[
0+P(Con)+P(Pro)=1
\]

所以 AU 内部的图立场向量不再经过第二次 softmax。

### 4. Initial 与 Graph 立场进行自适应融合

Initial 三分类概率：

\[
P_i^I=softmax(E_i^{init})
\]

图立场概率：

\[
P_i^G=G_i
\]

注意力打分和权重：

\[
s_i=V^T\tanh(W[P_i^I;P_i^G]+b)
\]

\[
[\beta_i^I,\beta_i^G]=softmax(s_i)
\]

最终融合概率：

\[
\boxed{
P_i^F=\beta_i^IP_i^I+\beta_i^GP_i^G
}
\]

当前最终输出保持 Initial CRF 的 `non/argument` 边界，只使用融合结果修改 argument token 的 Con/Pro 极性。

### 5. 所有损失等权直接相加

本次实验没有为各损失设置额外系数：

\[
\boxed{
\mathcal L=
\mathcal L_{FinalCRF}
+\mathcal L_{InitialCRF}
+\mathcal L_{OfficialToken}
+\mathcal L_{AU}
+\mathcal L_{Document}
}
\]

所有损失系数均为 1.0。

---

## 二、实验配置

| 配置项 | 数值 |
|---|---:|
| Batch size | 32 |
| 最大 Epoch | 30 |
| 实际停止 Epoch | 13 |
| Best Epoch | 8 |
| 学习率 | \(5\times10^{-6}\) |
| Weight decay | 0.02 |
| Warmup ratio | 0.1 |
| BERT dropout | 0.2 |
| GCN layers | 2 |
| GCN dropout | 0.2 |
| HetGAT layers | 2 |
| HetGAT heads | 1 |
| HetGAT dropout | 0.2 |
| AU semantic threshold | 0.5 |
| AU top-k | 3 |
| Syntax hops | 1 |
| 初始 non-emission bias | 0.0 |
| Early stopping patience | 5 |
| Checkpoint 指标 | Dev Official Token macro-F1 |

训练在 Epoch 8 得到最高 Dev Official Token F1，Epoch 9–13 连续五轮没有超过 Epoch 8，因此触发早停。

---

## 三、最佳 checkpoint 的 Dev 结果

| 指标 | Dev 数值 | 含义 |
|---|---:|---|
| Final / Official Token Macro-F1 | **0.7169** | 最终三分类 `non/con/pro` token macro-F1；checkpoint 选择指标 |
| Initial 3-class Token F1 | 0.7151 | Initial CRF 的三分类 token macro-F1 |
| Final − Initial | **+0.0018** | 图立场融合对最终 token F1 的影响 |
| Graph stance Token F1 | 0.5887 | HetGAT/Document 立场映射到全部 token 后的 macro-F1 |
| Fused stance Token F1 | 0.6902 | Initial 与 Graph 自适应融合后的 token macro-F1 |
| Official Segment F1 | 0.6216 | 官方 segment-level F1 |
| Official Sentence F1 | 0.7045 | 官方 sentence-level F1 |
| AU Token Precision | 0.6824 | token 是否属于 argument 的精确率 |
| AU Token Recall | 0.8704 | token 是否属于 argument 的召回率 |
| AU Token F1 | 0.7650 | 不要求严格起止位置的 AU token F1 |
| Initial Gold-AU stance Accuracy | 0.7895 | Initial CRF 在全部 342 个 Gold AU 上的多数票立场准确率 |
| Initial Gold-AU stance Macro-F1 | **0.8218** | Initial CRF 在全部 Gold AU 上的 Pro/Con macro-F1，不要求预测 span 匹配 |
| Gold-AU stance Accuracy | 0.7953 | 对全部 342 个 gold AU 进行立场判断的准确率 |
| Gold-AU stance Macro-F1 | 0.8228 | 对全部 gold AU 的 Pro/Con macro-F1 |
| Gold-AU Count | 342 | Dev gold AU 总数 |
| Matched-AU stance Accuracy | 0.8535 | 仅严格 span 匹配 AU 上的立场准确率 |
| Matched-AU stance Macro-F1 | 0.8531 | 仅严格 span 匹配 AU 上的立场 F1 |
| Matched-AU Count | 157 | 严格匹配并参与立场评价的 AU 数 |
| Strict AU Span F1 | 0.3179 | AU 起止位置完全一致时的 span F1 |
| Entity F1 | 0.2716 | 起止位置和立场均正确的严格 entity F1 |
| Document Stance Macro-F1 | 0.7047 | Document node 三分类立场 F1 |
| Final non-emission bias | 0.000012 | 从 0 初始化后学习得到的偏置，基本保持为 0 |

### Dev 损失

| 损失 | 数值 |
|---|---:|
| Final CRF | 23.5146 |
| Initial CRF | 22.9988 |
| Official Token | 0.8861 |
| AU stance | 0.5088 |
| Document stance | 1.8284 |
| Total | 49.7368 |

验证：

\[
23.5146+22.9988+0.8861+0.5088+1.8284
\approx49.7368
\]

---

## 四、最佳 checkpoint 的 Test 结果

| 指标 | Test 数值 | 含义 |
|---|---:|---|
| Final / Official Token Macro-F1 | **0.6838** | 当前实验最核心的 Test 指标 |
| Initial 3-class Token F1 | **0.6844** | Initial CRF 的三分类 token F1 |
| Final − Initial | **−0.0005** | 最终图融合轻微降低 Test token F1 |
| Graph stance Token F1 | 0.5816 | 单独图立场的 token macro-F1 |
| Fused stance Token F1 | 0.6707 | Initial 与 Graph 自适应融合后的 token F1 |
| Official Segment F1 | 0.5933 | 官方 segment-level F1 |
| Official Sentence F1 | 0.6998 | 官方 sentence-level F1 |
| AU Token Precision | 0.6640 | argument token 精确率 |
| AU Token Recall | 0.8144 | argument token 召回率 |
| AU Token F1 | **0.7315** | 不要求严格 AU 边界的 token-level AU F1 |
| All stance Token Accuracy | 0.5947 | Graph stance 在全部 token 上的三分类准确率 |
| All stance Token F1 | 0.5816 | Graph stance 在全部 token 上的三分类 macro-F1 |
| Argument stance Token Accuracy | 0.4145 | argument 相关 token 范围内的立场准确率 |
| Argument stance Token F1 | 0.3678 | argument 相关 token 范围内的立场 macro-F1 |
| Initial Gold-AU stance Accuracy | 0.7638 | Initial CRF 在全部 707 个 Gold AU 上的多数票立场准确率 |
| Initial Gold-AU stance Macro-F1 | **0.7994** | Initial CRF 在全部 Gold AU 上的 Pro/Con macro-F1，不要求预测 span 匹配 |
| Gold-AU stance Accuracy | 0.7680 | 对全部 707 个 gold AU 的立场准确率 |
| Gold-AU stance Macro-F1 | **0.7985** | 对全部 gold AU 的 Pro/Con macro-F1 |
| Gold-AU Count | 707 | Test gold AU 总数 |
| Matched-AU stance Accuracy | 0.9015 | 严格 span 匹配 AU 上的立场准确率 |
| Matched-AU stance Macro-F1 | 0.9014 | 严格 span 匹配 AU 上的立场 F1 |
| Matched-AU Count | 274 | 严格匹配并参与立场评价的 AU 数 |
| Strict AU Span Precision | 0.1984 | 严格 AU span precision |
| Strict AU Span Recall | 0.3918 | 严格 AU span recall |
| Strict AU Span F1 | **0.2634** | 严格 AU span F1 |
| Entity Precision | 0.1777 | 严格 entity precision |
| Entity Recall | 0.3508 | 严格 entity recall |
| Entity F1 | **0.2359** | AU 边界和立场同时正确的严格 F1 |
| Document Stance Macro-F1 | 0.6953 | Document stance macro-F1 |
| Final non-emission bias | 0.000012 | 学习后仍接近 0 |

### Test 损失

| 损失 | 数值 |
|---|---:|
| Final CRF | 25.0284 |
| Initial CRF | 24.3145 |
| Official Token | 0.9075 |
| AU stance | 0.2799 |
| Document stance | 1.7983 |
| Total | 52.3286 |

验证：

\[
25.0284+24.3145+0.9075+0.2799+1.7983
\approx52.3286
\]

---

## 五、训练过程

| Epoch | Dev Final Token F1 | Dev Initial Token F1 | Dev Graph F1 | Dev Fused F1 |
|---:|---:|---:|---:|---:|
| 1 | 0.2617 | 0.2620 | 0.2860 | 0.2622 |
| 2 | 0.5865 | 0.6019 | 0.5041 | 0.5608 |
| 3 | 0.6815 | 0.6802 | 0.5504 | 0.5802 |
| 4 | 0.6884 | 0.6886 | 0.5486 | 0.6018 |
| 5 | 0.6628 | 0.6632 | 0.5236 | 0.5888 |
| 6 | 0.6913 | 0.6899 | 0.5559 | 0.6466 |
| 7 | 0.7024 | 0.7016 | 0.5832 | 0.6787 |
| **8** | **0.7169** | **0.7151** | **0.5887** | **0.6902** |
| 9 | 0.6819 | 0.6808 | 0.5486 | 0.6575 |
| 10 | 0.7078 | 0.7073 | 0.5767 | 0.6932 |
| 11 | 0.6818 | 0.6820 | 0.5488 | 0.6643 |
| 12 | 0.6941 | 0.6936 | 0.5638 | 0.6793 |
| 13 | 0.6862 | 0.6852 | 0.5503 | 0.6737 |

Epoch 8 后 Dev 指标存在明显波动，并在连续五轮没有超过 0.7169 后停止。训练集在后期已达到约 0.92，而 Dev 维持在约 0.68–0.72，说明仍存在明显训练—验证泛化差距。

---

## 六、与旧实验结果比较

### 1. 与最近五分类实验 `run_20260811_093504` 比较

| Test 指标 | 旧实验 | 本次三分类实验 | 变化 |
|---|---:|---:|---:|
| Final Official Token F1 | 0.6935 | 0.6838 | **−0.0097** |
| Initial Token F1 | 0.6997 | 0.6844 | −0.0154 |
| Final − Initial | −0.0063 | −0.0005 | **改善 +0.0057** |
| Graph stance Token F1 | 0.6043 | 0.5816 | −0.0226 |
| Fused stance Token F1 | 0.6057 | 0.6707 | **+0.0651** |
| AU Token F1 | 0.7417 | 0.7315 | −0.0101 |
| Gold-AU stance F1 | 0.8158 | 0.7985 | −0.0173 |
| Strict AU Span F1 | 0.4329 | 0.2634 | −0.1695 |
| Entity F1 | 0.3818 | 0.2359 | −0.1459 |
| Document F1 | 0.7128 | 0.6953 | −0.0175 |

### 2. 与历史 Test 最好结果比较

历史最好实验：

```text
run_20260809_081411 / batch_size_32
```

其 Test Official Token F1：

\[
0.7040
\]

本次结果：

\[
0.6838
\]

变化：

\[
\boxed{0.6838-0.7040=-0.0201}
\]

因此，本次三分类版本没有超过历史最好 Test 结果。

---

## 七、结果分析

### 1. 三分类训练能够正常收敛

Epoch 1 时模型接近预测全 `non`，Dev F1 只有 0.2617；到 Epoch 3 已达到 0.6815，Epoch 8 达到 0.7169。这说明三分类 CRF、图网络和等权损失能够完成端到端训练，没有发生持续的 `non` 类塌缩。

### 2. 图融合对最终 token 的作用已经接近中性

Dev：

\[
0.7169-0.7151=+0.0018
\]

Test：

\[
0.6838-0.6844=-0.0005
\]

相比旧实验 Test 中约 −0.0063 的下降，本次最终图融合对 token 结果的破坏已经显著减小，但尚未形成稳定的 Test 增益。

### 3. 自适应融合明显改善了单独 Graph stance

Test：

\[
0.6707-0.5816=+0.0891
\]

说明 attention fusion 确实能够自动更多地依赖质量更高的 Initial token 分支。但是融合结果仍低于 Initial：

\[
0.6707<0.6844
\]

因此 Graph 分支目前仍然是融合性能的限制因素。

### 4. 删除 B/I 后，严格边界指标下降符合预期

三分类只保留 `non/con/pro`，不再显式标记 AU 起点。因此：

- AU Token F1 仍有 0.7315，说明 argument token 覆盖能力尚可；
- Strict AU Span F1 只有 0.2634；
- Entity F1 只有 0.2359。

这并不完全代表 token 识别失败，而是三分类表示无法区分相邻同立场 AU，导致严格 span 和 entity 指标天然受损。

### 5. Gold-AU 立场识别仍然较好

对全部 707 个 Test gold AU：

\[
F1=0.7985,\qquad Accuracy=0.7680
\]

这说明 AU/HetGAT 立场推理具备有效信息。严格匹配 AU 上的 F1 高达 0.9014，但它只统计 274 个成功匹配的 AU，存在明显的容易样本筛选效应，不能代替全部 Gold-AU 指标。

Initial CRF 在相同 707 个 Test Gold AU 上的立场结果为：

\[
F1_{Initial\ Gold-AU}=0.7994
\]

图立场结果为：

\[
F1_{Graph\ Gold-AU}=0.7985
\]

两者差值：

\[
0.7985-0.7994=-0.0010
\]

因此在本次 Test 上，HetGAT 后的 Gold-AU 立场没有超过 Initial CRF，二者基本持平且图分支略低。

### 6. Dev 与 Test 存在差距

\[
0.7169-0.6838=0.0331
\]

Dev 最优较高，但对应 Test 没有同步提升，说明仍存在泛化差距。训练后期 Train F1 超过 0.91，也进一步表明模型存在过拟合倾向。

---

## 八、最终结论

本次实验正确完成了以下目标：

1. 全程三分类 `non/con/pro`；
2. 删除 HetGAT 的 \(\gamma_r\)；
3. AU 图立场直接使用 \([0,P(Con),P(Pro)]\)；
4. Initial 与 Graph 采用自适应注意力融合；
5. 五项损失全部以系数 1.0 直接相加；
6. 使用 Dev Official Token macro-F1 进行 checkpoint 选择和早停。

最终核心结果：

\[
\boxed{
\text{Dev Final Token F1}=0.7169
}
\]

\[
\boxed{
\text{Test Final Token F1}=0.6838
}
\]

\[
\boxed{
\text{Test Initial Token F1}=0.6844
}
\]

\[
\boxed{
\text{Test Final-Initial}=-0.0005
}
\]

总体来看，三分类版本训练正常，图反馈对最终 token 的负面影响已经缩小到接近零，但 Test Final Token F1 仍低于历史最好结果 0.7040。下一步应优先提高 Graph stance 的全 token 表现和跨 Dev/Test 的泛化能力，而不是继续增强严格 AU span 指标，因为三分类结构已经主动放弃了 B/I 边界信息。
