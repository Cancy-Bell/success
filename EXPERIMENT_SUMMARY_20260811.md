# 2026-08-11 两次实验结果总结

本文档总结最近两次实验的模型改动、Dev/Test 指标结果以及结论。两次实验的 checkpoint 选择指标均为：

\[
\text{Dev Official Token Macro-F1}
\]

也就是官方 AURC 三分类 token-level macro-F1。

---

## 一、实验目录

| 实验 | 目录 | Best Epoch | 主要变化 |
|---|---|---:|---|
| 实验一 | `F:\AURC-master\models\batch_size_comparison\run_20260811_075807\batch_size_32` | 8 | 加入 Initial stance 与 HetGAT stance 的 attention fusion，并将 fused stance bias 注入 final BIO emission |
| 实验二 | `F:\AURC-master\models\batch_size_comparison\run_20260811_093504\batch_size_32` | 15 | 保留 graph/fused stance 诊断输出，但取消 fused stance bias 对 final BIO emission 的直接注入 |

---

## 二、实验一：Attention fused stance 直接注入 final BIO

### 1. 模型改动

实验一在原有闭环结构基础上加入：

\[
z_t^I = \text{Initial official logits}
\]

\[
z_t^G = \text{Graph / HetGAT official logits}
\]

然后计算注意力融合：

\[
s_t = V^T \tanh(W[z_t^I;z_t^G] + b)
\]

\[
\beta_t = softmax(s_t)
\]

\[
z_t^F = \beta_{t,I}z_t^I + \beta_{t,G}z_t^G
\]

再将三分类 fused stance logits 映射回五分类 BIO 空间：

\[
z_t^F \rightarrow [O,B\text{-}Pro,I\text{-}Pro,B\text{-}Con,I\text{-}Con]
\]

最终：

\[
E_t^{final}=E_t^{initial}+g_tR_t+\lambda z_t^F
\]

其中：

\[
\lambda = 0.1
\]

### 2. Dev 指标

| 指标 | 数值 | 指标含义 |
|---|---:|---|
| Official Token Macro-F1 / Final 3-class Token F1 | 0.7031 | 最终输出 `final_bio` 折算为官方三分类 `non/con/pro` 后，在 token 级计算 macro-F1；这是 checkpoint 选择的核心指标 |
| Initial 3-class Token F1 | 0.6999 | 初始 CRF 输出 `initial_crf_bio` 折算为官方三分类后的 token-level macro-F1，用来看前半段 BIO 序列标注能力 |
| Final - Initial Delta | +0.0032 | Final 3-class Token F1 减去 Initial 3-class Token F1，用来看图反馈/融合是否提升最终 token 结果 |
| Graph stance Token F1 | 0.6076 | HetGAT 后 AU/Document 立场结果映射回 token 后的三分类 macro-F1，用来看图立场推理本身质量 |
| Fused stance Token F1 | 0.6453 | Initial stance 与 Graph stance 经过 attention fusion 后映射回 token 的三分类 macro-F1 |
| Official Segment F1 | 0.6608 | 官方 AURC segment-level F1，按连续片段级别评估 `non/con/pro` 预测 |
| Official Sentence F1 | 0.7068 | 官方 AURC sentence-level F1，按句子整体标签评估 |
| All stance Token Acc | 0.6324 | Graph stance 映射到所有 token 后的三分类准确率，包含 `non/con/pro` 全部 token |
| All stance Token F1 | 0.6076 | Graph stance 映射到所有 token 后的三分类 macro-F1，包含 `non/con/pro` 全部 token |
| Argument stance Token Acc | 0.4229 | 只在 gold 或 graph 判断为 argument 的 token 上计算 Pro/Con/Non 准确率，重点看论据相关 token 的立场质量 |
| Argument stance Token F1 | 0.3756 | 只在 argument 相关 token 上计算 macro-F1，排除大量纯 non token 的掩盖效应 |
| Argument stance Token Count | 11597 | 参与 Argument stance 指标计算的 token 数量 |
| Gold-AU stance Acc | 0.7515 | 对每个 gold AU 判断其立场是否预测正确的准确率；不要求 predicted AU span 严格匹配 |
| Gold-AU stance F1 | 0.7908 | 对所有 gold AU 的 Pro/Con 立场预测计算 macro-F1；这是你要求的“test 一共 m 条 AU，看 m 条里立场识别对多少”的指标 |
| Gold-AU Count | 342 | Dev 集 gold AU 总数 |
| Strict AU Span Precision | 0.4564 | 严格 AU span 匹配下的 precision，要求预测 AU 起止位置完全正确 |
| Strict AU Span Recall | 0.5351 | 严格 AU span 匹配下的 recall，要求 gold AU 被完整预测出来 |
| Strict AU Span F1 | 0.4926 | 严格 AU span 级别 F1；这是辅助指标，不是你后续要求的 token 级 AU 识别 |
| Initial AU Token Precision | 0.6631 | Initial BIO 在 token 级判断“是否属于 AU”的 precision，不要求 AU 起止位置严格匹配 |
| Initial AU Token Recall | 0.8580 | Initial BIO 在 token 级判断 AU 覆盖范围的 recall |
| Initial AU Token F1 | 0.7481 | Initial BIO 的 token 级 AU 识别 F1，只看是否为 argument，不看 Pro/Con |
| Final AU Token Precision | 0.6631 | Final BIO 在 token 级判断“是否属于 AU”的 precision |
| Final AU Token Recall | 0.8580 | Final BIO 在 token 级判断 AU 覆盖范围的 recall |
| Final AU Token F1 | 0.7481 | Final BIO 的 token 级 AU 识别 F1，只看 AU 覆盖，不看严格 span |
| Matched-AU stance Acc | 0.8361 | 旧版严格 matched AU stance accuracy，只在预测 span 与 gold span 完全匹配的 AU 上评估立场 |
| Matched-AU stance F1 | 0.8361 | 旧版严格 matched AU stance macro-F1，只在严格匹配 AU 上计算 |
| Matched-AU Count | 183 | 严格匹配成功并参与 Matched-AU stance 计算的 AU 数量 |
| Entity Precision | 0.3865 | Entity-level precision，同时考虑 AU 边界和 Pro/Con 类型 |
| Entity Recall | 0.4532 | Entity-level recall，同时考虑 AU 边界和 Pro/Con 类型 |
| Entity F1 | 0.4172 | Entity-level F1，评估完整论据单元抽取与立场类型是否正确 |
| Document Stance Macro-F1 | 0.7124 | Document node 的句子/文档级 stance 分类 macro-F1 |
| Initial 5-class BIO Token F1 | 0.6159 | Initial BIO 五分类 `O/B-Pro/I-Pro/B-Con/I-Con` token macro-F1 |
| Final 5-class BIO Token F1 | 0.6180 | Final BIO 五分类 token macro-F1 |
| 5-class BIO Delta | +0.0021 | Final 5-class BIO F1 减去 Initial 5-class BIO F1 |
| Total Loss | 12.6167 | 总训练/评估 loss |
| BIO Combined Loss | 10.8082 | BIO 相关组合 loss，包括 FinalCRF、InitialCRF 辅助项和 OfficialTokenCE |
| Final CRF Loss | 9.5026 | Final BIO-CRF 的负对数似然 loss |
| Initial CRF Loss | 2.9840 | Initial BIO-CRF 的辅助负对数似然 loss |
| Official Token CE Loss | 0.8210 | 官方三分类 token CE 辅助 loss |
| AU Loss | 0.3323 | AU stance 分类 loss |
| Document Loss | 1.4762 | Document stance 分类 loss |
| Final O-bias | 0.3254 | Final emission 中 O 类的可学习偏置，用于校准 non/O 预测倾向 |

### 3. Test 指标

| 指标 | 数值 | 指标含义 |
|---|---:|---|
| Official Token Macro-F1 / Final 3-class Token F1 | 0.6736 | 最终输出 `final_bio` 折算为官方三分类后的 token-level macro-F1 |
| Initial 3-class Token F1 | 0.6789 | 初始 CRF 输出折算为官方三分类后的 token-level macro-F1 |
| Final - Initial Delta | -0.0053 | Final 相对 Initial 的变化，负数说明最终图反馈/融合拉低 token F1 |
| Graph stance Token F1 | 0.5900 | HetGAT 立场结果映射回 token 后的三分类 macro-F1 |
| Fused stance Token F1 | 0.6272 | Initial stance 与 Graph stance attention fusion 后的三分类 token macro-F1 |
| Official Segment F1 | 0.6389 | 官方 segment-level F1 |
| Official Sentence F1 | 0.6951 | 官方 sentence-level F1 |
| All stance Token Acc | 0.6073 | Graph stance 在所有 token 上的三分类准确率 |
| All stance Token F1 | 0.5900 | Graph stance 在所有 token 上的三分类 macro-F1 |
| Argument stance Token Acc | 0.4152 | argument 相关 token 范围内的 stance 准确率 |
| Argument stance Token F1 | 0.3697 | argument 相关 token 范围内的 stance macro-F1 |
| Argument stance Token Count | 24212 | 参与 Argument stance 指标计算的 token 数量 |
| Gold-AU stance Acc | 0.7412 | 对所有 gold AU 判断 Pro/Con 立场是否正确的准确率 |
| Gold-AU stance F1 | 0.7845 | 对所有 gold AU 的 Pro/Con 立场预测 macro-F1 |
| Gold-AU Count | 707 | Test 集 gold AU 总数 |
| Strict AU Span Precision | 0.4082 | 严格 AU span 匹配 precision |
| Strict AU Span Recall | 0.4781 | 严格 AU span 匹配 recall |
| Strict AU Span F1 | 0.4404 | 严格 AU span 级别 F1 |
| Initial AU Token Precision | 0.6405 | Initial BIO 的 token 级 AU precision |
| Initial AU Token Recall | 0.8396 | Initial BIO 的 token 级 AU recall |
| Initial AU Token F1 | 0.7267 | Initial BIO 的 token 级 AU F1 |
| Final AU Token Precision | 0.6406 | Final BIO 的 token 级 AU precision |
| Final AU Token Recall | 0.8397 | Final BIO 的 token 级 AU recall |
| Final AU Token F1 | 0.7267 | Final BIO 的 token 级 AU F1 |
| Matched-AU stance Acc | 0.8665 | 旧版严格 matched AU stance accuracy |
| Matched-AU stance F1 | 0.8664 | 旧版严格 matched AU stance macro-F1 |
| Matched-AU Count | 337 | 严格匹配成功并参与 Matched-AU stance 计算的 AU 数量 |
| Entity Precision | 0.3551 | Entity-level precision |
| Entity Recall | 0.4158 | Entity-level recall |
| Entity F1 | 0.3831 | Entity-level F1 |
| Document Stance Macro-F1 | 0.6915 | Document stance macro-F1 |
| Initial 5-class BIO Token F1 | 0.5951 | Initial 五分类 BIO token macro-F1 |
| Final 5-class BIO Token F1 | 0.5898 | Final 五分类 BIO token macro-F1 |
| 5-class BIO Delta | -0.0052 | Final 五分类 BIO F1 相对 Initial 的变化 |
| Total Loss | 35.5130 | 总 loss |
| BIO Combined Loss | 33.7103 | BIO 相关组合 loss |
| Final CRF Loss | 27.3377 | Final BIO-CRF loss |
| Initial CRF Loss | 19.7882 | Initial BIO-CRF 辅助 loss |
| Official Token CE Loss | 0.8722 | 官方三分类 token CE 辅助 loss |
| AU Loss | 0.2328 | AU stance loss |
| Document Loss | 1.5700 | Document stance loss |
| Final O-bias | 0.3254 | Final O 类可学习偏置 |

### 4. 实验一结论

实验一说明 attention fusion 本身能增强 graph stance：

\[
\text{Fused stance Token F1}=0.6272
>
\text{Graph stance Token F1}=0.5900
\]

但是 fused stance 仍明显低于 Initial token 分支：

\[
0.6272 < 0.6789
\]

因此当 fused stance 被直接注入 final BIO emission 后，Test final token F1 反而下降：

\[
0.6736 - 0.6789 = -0.0053
\]

结论：直接将 fused stance logits 注入完整 BIO emission 不稳定，容易干扰 O / B / I 边界和 Pro / Con 极性。

---

## 三、实验二：取消 fused stance 对 final BIO 的直接注入

### 1. 模型改动

实验二保留以下诊断输出：

- Graph stance Token F1
- Fused stance Token F1
- `graph_official_probs`
- `fused_official_probs`
- `stance_fusion_weights`

但是取消 fused stance bias 对 final BIO emission 的直接影响。

原实验一：

```python
final_emissions = (
    initial_emissions
    + feedback_gates * reasoning_correction
    + stance_fusion_to_bio_scale * fused_bio_stance_bias
)
```

实验二改为：

```python
final_emissions = initial_emissions + feedback_gates * reasoning_correction
```

也就是说，attention fused stance 只作为诊断信号输出，不直接参与 final BIO 解码。

### 2. Dev 指标

| 指标 | 数值 | 指标含义 |
|---|---:|---|
| Official Token Macro-F1 / Final 3-class Token F1 | 0.7125 | 最终输出折算为官方三分类后的 token-level macro-F1 |
| Initial 3-class Token F1 | 0.7172 | Initial CRF 折算为官方三分类后的 token-level macro-F1 |
| Final - Initial Delta | -0.0047 | Final 相对 Initial 的变化 |
| Graph stance Token F1 | 0.6019 | HetGAT 立场结果映射回 token 后的三分类 macro-F1 |
| Fused stance Token F1 | 0.6027 | Initial stance 与 Graph stance 融合后的三分类 token macro-F1 |
| Official Segment F1 | 0.6639 | 官方 segment-level F1 |
| Official Sentence F1 | 0.7197 | 官方 sentence-level F1 |
| All stance Token Acc | 0.6196 | Graph stance 在所有 token 上的三分类准确率 |
| All stance Token F1 | 0.6019 | Graph stance 在所有 token 上的三分类 macro-F1 |
| Argument stance Token Acc | 0.4257 | argument 相关 token 上的 stance 准确率 |
| Argument stance Token F1 | 0.3777 | argument 相关 token 上的 stance macro-F1 |
| Argument stance Token Count | 12058 | 参与 Argument stance 指标计算的 token 数量 |
| Gold-AU stance Acc | 0.7895 | 所有 gold AU 的 Pro/Con 立场准确率 |
| Gold-AU stance F1 | 0.8157 | 所有 gold AU 的 Pro/Con 立场 macro-F1 |
| Gold-AU Count | 342 | Dev 集 gold AU 总数 |
| Strict AU Span Precision | 0.4515 | 严格 AU span 匹配 precision |
| Strict AU Span Recall | 0.5848 | 严格 AU span 匹配 recall |
| Strict AU Span F1 | 0.5096 | 严格 AU span F1 |
| Initial AU Token Precision | 0.6720 | Initial token 级 AU precision |
| Initial AU Token Recall | 0.8911 | Initial token 级 AU recall |
| Initial AU Token F1 | 0.7662 | Initial token 级 AU F1 |
| Final AU Token Precision | 0.6610 | Final token 级 AU precision |
| Final AU Token Recall | 0.9031 | Final token 级 AU recall |
| Final AU Token F1 | 0.7633 | Final token 级 AU F1 |
| Matched-AU stance Acc | 0.8622 | 旧版严格匹配 AU 上的 stance accuracy |
| Matched-AU stance F1 | 0.8622 | 旧版严格匹配 AU 上的 stance macro-F1 |
| Matched-AU Count | 196 | 严格匹配 AU 数量 |
| Entity Precision | 0.3860 | Entity-level precision |
| Entity Recall | 0.5000 | Entity-level recall |
| Entity F1 | 0.4357 | Entity-level F1 |
| Document Stance Macro-F1 | 0.7080 | Document stance macro-F1 |
| Initial 5-class BIO Token F1 | 0.6392 | Initial 五分类 BIO token macro-F1 |
| Final 5-class BIO Token F1 | 0.6370 | Final 五分类 BIO token macro-F1 |
| 5-class BIO Delta | -0.0023 | Final 五分类 BIO F1 相对 Initial 的变化 |
| Total Loss | 10.5385 | 总 loss |
| BIO Combined Loss | 7.3869 | BIO 相关组合 loss |
| Final CRF Loss | 5.4927 | Final BIO-CRF loss |
| Initial CRF Loss | 4.8654 | Initial BIO-CRF 辅助 loss |
| Official Token CE Loss | 0.8690 | 官方三分类 token CE 辅助 loss |
| AU Loss | 0.5379 | AU stance loss |
| Document Loss | 2.6138 | Document stance loss |
| Final O-bias | 0.3249 | Final O 类可学习偏置 |

### 3. Test 指标

| 指标 | 数值 | 指标含义 |
|---|---:|---|
| Official Token Macro-F1 / Final 3-class Token F1 | 0.6935 | 最终输出折算为官方三分类后的 token-level macro-F1 |
| Initial 3-class Token F1 | 0.6997 | Initial CRF 折算为官方三分类后的 token-level macro-F1 |
| Final - Initial Delta | -0.0063 | Final 相对 Initial 的变化，负数说明 feedback 拉低 token F1 |
| Graph stance Token F1 | 0.6043 | HetGAT 立场结果映射回 token 后的三分类 macro-F1 |
| Fused stance Token F1 | 0.6057 | Initial stance 与 Graph stance 融合后的三分类 token macro-F1 |
| Official Segment F1 | 0.6488 | 官方 segment-level F1 |
| Official Sentence F1 | 0.7146 | 官方 sentence-level F1 |
| All stance Token Acc | 0.6186 | Graph stance 在所有 token 上的三分类准确率 |
| All stance Token F1 | 0.6043 | Graph stance 在所有 token 上的三分类 macro-F1 |
| Argument stance Token Acc | 0.4328 | argument 相关 token 上的 stance 准确率 |
| Argument stance Token F1 | 0.3829 | argument 相关 token 上的 stance macro-F1 |
| Argument stance Token Count | 24242 | 参与 Argument stance 指标计算的 token 数量 |
| Gold-AU stance Acc | 0.7808 | 所有 gold AU 的 Pro/Con 立场准确率 |
| Gold-AU stance F1 | 0.8158 | 所有 gold AU 的 Pro/Con 立场 macro-F1 |
| Gold-AU Count | 707 | Test 集 gold AU 总数 |
| Strict AU Span Precision | 0.3873 | 严格 AU span 匹配 precision |
| Strict AU Span Recall | 0.4908 | 严格 AU span 匹配 recall |
| Strict AU Span F1 | 0.4329 | 严格 AU span F1 |
| Initial AU Token Precision | 0.6662 | Initial token 级 AU precision |
| Initial AU Token Recall | 0.8465 | Initial token 级 AU recall |
| Initial AU Token F1 | 0.7456 | Initial token 级 AU F1 |
| Final AU Token Precision | 0.6536 | Final token 级 AU precision |
| Final AU Token Recall | 0.8572 | Final token 级 AU recall |
| Final AU Token F1 | 0.7417 | Final token 级 AU F1 |
| Matched-AU stance Acc | 0.8717 | 旧版严格匹配 AU 上的 stance accuracy |
| Matched-AU stance F1 | 0.8714 | 旧版严格匹配 AU 上的 stance macro-F1 |
| Matched-AU Count | 343 | 严格匹配 AU 数量 |
| Entity Precision | 0.3415 | Entity-level precision |
| Entity Recall | 0.4328 | Entity-level recall |
| Entity F1 | 0.3818 | Entity-level F1 |
| Document Stance Macro-F1 | 0.7128 | Document stance macro-F1 |
| Initial 5-class BIO Token F1 | 0.6103 | Initial 五分类 BIO token macro-F1 |
| Final 5-class BIO Token F1 | 0.6092 | Final 五分类 BIO token macro-F1 |
| 5-class BIO Delta | -0.0011 | Final 五分类 BIO F1 相对 Initial 的变化 |
| Total Loss | 31.9721 | 总 loss |
| BIO Combined Loss | 29.0988 | BIO 相关组合 loss |
| Final CRF Loss | 22.1820 | Final BIO-CRF loss |
| Initial CRF Loss | 21.5800 | Initial BIO-CRF 辅助 loss |
| Official Token CE Loss | 0.8858 | 官方三分类 token CE 辅助 loss |
| AU Loss | 0.3517 | AU stance loss |
| Document Loss | 2.5215 | Document stance loss |
| Final O-bias | 0.3249 | Final O 类可学习偏置 |

### 4. 实验二结论

实验二中，Initial token 分支表现明显更好：

\[
\text{Test Initial 3-class Token F1}=0.6997
\]

已经超过官方 AURC 参考结果约：

\[
0.696
\]

说明：

\[
\text{Dual-BERT + Dependency GCN + Semantic-Syntax Fusion + Initial BIO-CRF}
\]

这条前半段对 token-level 论据识别是有效的。

但是 final feedback 仍然降低 token F1：

\[
0.6935 - 0.6997 = -0.0063
\]

这说明当前：

\[
\text{HetGAT / AU stance / Document stance feedback}
\]

回写到 token BIO 后仍不稳定。

---

## 四、两次实验 Test 指标对比

| 指标 | 实验一：fused stance 注入 BIO | 实验二：取消 fused stance 注入 | 变化 | 指标含义 |
|---|---:|---:|---:|---|
| Best Epoch | 8 | 15 | +7 | Dev Official Token F1 最高时对应的 epoch |
| Dev Official Token F1 | 0.7031 | 0.7125 | +0.0094 | Dev 集最终三分类 token macro-F1，也是 best checkpoint 的选择依据 |
| Test Official Token F1 / Final 3-class Token F1 | 0.6736 | 0.6935 | +0.0199 | Test 集最终三分类 token macro-F1，最核心测试指标 |
| Test Initial 3-class Token F1 | 0.6789 | 0.6997 | +0.0208 | Test 集 Initial CRF 三分类 token macro-F1，用来看前半段模型能力 |
| Final - Initial Delta | -0.0053 | -0.0063 | -0.0010 | Final 与 Initial 的差值，用来看 feedback/fusion 是否真正带来增益 |
| Graph stance Token F1 | 0.5900 | 0.6043 | +0.0143 | HetGAT stance 映射回 token 后的三分类 macro-F1 |
| Fused stance Token F1 | 0.6272 | 0.6057 | -0.0215 | Initial stance 与 Graph stance 融合后的 token macro-F1 |
| Official Segment F1 | 0.6389 | 0.6488 | +0.0098 | 官方 segment-level F1 |
| Official Sentence F1 | 0.6951 | 0.7146 | +0.0195 | 官方 sentence-level F1 |
| AU Token F1 | 0.7267 | 0.7417 | +0.0149 | token 级 AU 识别 F1，只看是否为论据，不要求严格 span |
| Gold-AU stance F1 | 0.7845 | 0.8158 | +0.0313 | 对所有 gold AU 的 Pro/Con 立场识别 macro-F1，不要求 predicted AU span 严格匹配 |
| Strict AU Span F1 | 0.4404 | 0.4329 | -0.0075 | 严格 AU 起止位置完全匹配的 span-level F1 |
| Entity F1 | 0.3831 | 0.3818 | -0.0013 | 同时考虑 AU 边界和立场类型的 entity-level F1 |
| Document F1 | 0.6915 | 0.7128 | +0.0213 | Document stance macro-F1 |
| Initial 5-class BIO Token F1 | 0.5951 | 0.6103 | +0.0152 | Initial 五分类 BIO token macro-F1 |
| Final 5-class BIO Token F1 | 0.5898 | 0.6092 | +0.0194 | Final 五分类 BIO token macro-F1 |
| Final O-bias | 0.3254 | 0.3249 | -0.0004 | Final O 类可学习偏置 |

---

## 五、总体结论

### 1. 句法增强的 Initial BIO-CRF 是有效的

实验二中：

\[
\text{Test Initial 3-class Token F1}=0.6997
\]

已经高于官方参考结果约：

\[
0.696
\]

因此目前最稳定、最有价值的部分是：

\[
\boxed{
\text{Dual-BERT + Dependency GCN + Semantic-Syntax Fusion + Initial BIO-CRF}
}
\]

### 2. HetGAT stance 信息目前适合做辅助分析，但不适合直接强回写 token

两次实验都显示：

\[
\text{Final 3-class Token F1} < \text{Initial 3-class Token F1}
\]

实验一：

\[
0.6736 < 0.6789
\]

实验二：

\[
0.6935 < 0.6997
\]

这说明图推理反馈目前仍会伤害 final BIO token prediction。

### 3. Attention fusion 不是完全无效，但直接注入 BIO 的方式有风险

实验一中：

\[
\text{Fused stance Token F1}=0.6272
>
\text{Graph stance Token F1}=0.5900
\]

说明 Initial stance 与 Graph stance 的 attention fusion 能改善 graph stance 本身。

但它仍然低于 Initial token 分支，因此直接注入 BIO emission 会伤害最终 token F1。

### 4. 下一步建议

下一步如果继续使用 attention fusion，更推荐采用：

\[
\boxed{
\text{Boundary frozen / stance-only fusion}
}
\]

也就是：

- Initial CRF 决定 `O / B / I` 边界；
- Initial stance 与 HetGAT stance 通过 attention 自适应融合；
- 融合结果只修正 `Pro / Con` 极性；
- 不让 HetGAT stance 改变 `O` 与 argument boundary。

这样可以避免 graph stance 质量不足时破坏 token 边界。
