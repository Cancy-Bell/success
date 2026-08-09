# AURC End-to-End Graph Model Handoff

Last updated: 2026-08-08

This document summarizes the current code state, model flow, training objective,
metrics, observed results, and recommended next steps for the modified official
AURC source project.

## 1. Current Implementation Status

The project has been modified directly on top of the official AURC codebase.
The current main entry point is:

```text
src/run_AURC_token.py
```

The current model implementation is:

```text
src/models.py
```

The current experiment launcher is:

```text
run_batch_size_experiments.py
```

The current model is no longer the old two-stage epoch-16 graph-activation
model. It is now a full end-to-end feedback model trained from epoch 1:

```text
Dual-BERT
  -> dependency alignment
  -> Dependency GCN
  -> Semantic-Syntax Fusion
  -> Initial BIO-CRF
  -> AU/document representation
  -> batch-level HetGAT
  -> AU stance + document stance
  -> stance-guided token feedback
  -> Final BIO-CRF
```

The current early stopping and best checkpoint selection criterion is:

```text
Dev Official Token macro-F1
```

The current checkpoint file is:

```text
checkpoints/best_token_model.pt
```

## 2. Label Space

The token-level BIO label space has 5 classes:

```text
0: O
1: B-Pro
2: I-Pro
3: B-Con
4: I-Con
```

The AU stance classifier is binary:

```text
Pro / Con
```

The document stance classifier is 3-way:

```text
non / con / pro
```

The official AURC token metric collapses the 5-class BIO labels into 3 classes:

```text
O                 -> non
B-Con / I-Con     -> con
B-Pro / I-Pro     -> pro
```

The current checkpoint selection uses this collapsed official token macro-F1,
not Entity F1 and not 5-class BIO macro-F1.

## 3. Full Model Flow and Formulas

### 3.1 Dual-BERT Encoding

The model uses two independent BERT encoders.

The joint encoder receives topic and sentence together:

```text
[CLS] Topic [SEP] Sentence [SEP]
```

It produces contextual token states:

```math
H^B = [h_1^B, h_2^B, \ldots, h_L^B]
```

The topic-only encoder receives:

```text
[CLS] Topic [SEP]
```

The topic representation is the topic encoder CLS vector:

```math
h_T = H_T[CLS]
```

The previous topic masked mean pooling has been removed.

### 3.2 Dependency Alignment

spaCy produces a word-level dependency adjacency matrix:

```math
A^{spacy} \in \mathbb{R}^{N \times N}
```

Because BERT works on WordPieces, the word-level dependency matrix is mapped
into WordPiece space by an alignment matrix:

```math
M \in \mathbb{R}^{L \times N}
```

The final WordPiece dependency matrix is:

```math
A' = M A^{spacy} M^T
```

This is still part of the forward path from the first epoch.

### 3.3 Dependency GCN

The Dependency GCN propagates syntactic information over the WordPiece graph:

```math
H^{syn} = GCN(H^B, A')
```

In code this is implemented by `DependencyGCN`.

### 3.4 Semantic-Syntax Fusion

The previous token-topic fusion has been removed. The current fusion is between
BERT semantic states and dependency-enhanced syntactic states:

```math
Z^{ss} = Fusion(H^B, H^{syn})
```

The fusion is gated. Conceptually:

```math
C = GELU(W_c[H^B; H^{syn}; H^B - H^{syn}; H^B \odot H^{syn}])
```

```math
G = \sigma(W_g[H^B; H^{syn}])
```

```math
Z^{ss} = LayerNorm(G \odot C + (1-G) \odot H^B)
```

Then the final token feature is:

```math
H = Dropout(GELU(Z^{ss}))
```

There is no extra outer residual:

```math
H \ne H^B + Z^{ss}
```

This avoids repeatedly strengthening BERT semantic states after fusion.

### 3.5 Initial BIO-CRF

The initial BIO classifier produces 5-class emissions:

```math
E^{(0)} = W_{BIO}H + b_{BIO}
```

Then CRF decoding gives the initial BIO sequence:

```math
Y^{initial} = CRFDecode(E^{(0)})
```

This is called the initial BIO prediction, not the final prediction.

### 3.6 AU Extraction for Graph Construction

During training, graph AU spans are extracted from gold BIO labels:

```math
Y^{graph} = Y^{gold}
```

During evaluation and testing, graph AU spans are extracted from the initial
predicted BIO labels:

```math
Y^{graph} = Y^{initial}
```

This means training uses reliable supervised AU nodes, while evaluation follows
the real inference pipeline.

### 3.7 Topic-Aware AU Representation

AU representation no longer uses mean pooling.

For an AU span:

```math
AU_k = [s_k, e_k)
```

Construct a token mask:

```math
m_i^{(k)} =
\begin{cases}
1, & i \in AU_k \\
0, & otherwise
\end{cases}
```

Use the topic representation as query:

```math
q_T = W_Q h_T
```

For each token:

```math
k_i = W_K h_i,\quad v_i = W_V h_i
```

The masked attention score is:

```math
e_i^{(k)} =
\begin{cases}
\frac{q_T^T k_i}{\sqrt d}, & i \in AU_k \\
-\infty, & otherwise
\end{cases}
```

The AU attention weights are:

```math
\alpha_i^{(k)} = softmax(e_i^{(k)})
```

The AU representation is:

```math
h_{AU_k} = \sum_i \alpha_i^{(k)} v_i
```

### 3.8 Topic-Aware Document Representation

Document representation also does not use mean pooling.

It uses full-token attention:

```math
e_i^D = \frac{q_T^T k_i}{\sqrt d}
```

```math
\alpha_i^D = softmax(e_i^D)
```

```math
h_D = \sum_i \alpha_i^D v_i
```

All valid sentence tokens participate.

### 3.9 Batch-Level HetGAT Graph

The current graph is one graph per batch, not one graph per sample.

The node set is:

```math
V = V_{AU} \cup V_D
```

There is no topic node. Topic information is already injected through topic-only
BERT, AU attention, document attention, and AU stance classification.

The graph contains:

```text
AU-AU edges inside the same document
AU-Document edges inside the same document
Document-Document edges between documents with the same topic
Self edges
```

AU-AU semantic and syntax evidence are merged into one AU-AU relation:

```math
A^{AU} = A^{sem} + A^{syn}
```

They are not treated as separate HetGAT relation types.

The current default HetGAT head count is:

```text
hetgat_heads = 1
```

The graph update is:

```math
H^G = HetGAT(V, E)
```

The output gives graph-enhanced AU and document node states:

```math
h_{AU_k}^G,\quad h_D^G
```

### 3.10 AU Stance Prediction

There is only one AU stance classifier now. The old sequence-vs-graph stance
dual-branch and dynamic stance fusion have been removed.

The final AU stance classifier uses the HetGAT-updated AU representation and
the topic representation:

```math
z_k^{AU} = W_{AU}[h_{AU_k}^G; h_T] + b_{AU}
```

```math
p_k^{AU} = softmax(z_k^{AU})
```

where:

```math
p_k^{AU} = [P(Pro), P(Con)]
```

The AU stance loss is:

```math
\mathcal{L}_{AU} = CE(z^{AU}, y^{AU})
```

### 3.11 Document Stance Prediction

The document classifier uses the HetGAT-updated document node:

```math
z_D = W_D h_D^G + b_D
```

```math
p_D = softmax(z_D)
```

where:

```math
p_D = [P(non), P(con), P(pro)]
```

The document stance loss is:

```math
\mathcal{L}_D = CE(z_D, y_D)
```

### 3.12 Stance-Guided Token Feedback

This is the current model's main closed loop:

```text
Token -> AU -> Document/Graph -> Token
```

For a token inside AU k, AU stance probabilities are mapped back to token-level
5-class correction evidence:

```math
c_i^{AU} = [0,\ p_k^{Pro},\ p_k^{Pro},\ p_k^{Con},\ p_k^{Con}]
```

The class order is:

```text
[O, B-Pro, I-Pro, B-Con, I-Con]
```

Document stance is projected into a 5-class token prior:

```math
c_i^D = W_D^{token}p_D + b_D^{token}
```

The feedback gate is:

```math
g_i = \sigma(W_g[h_i; c_i^{AU}; c_i^D] + b_g)
```

The reasoning correction is:

```math
R_i = W_r[h_i; c_i^{AU}; c_i^D] + b_r
```

The final emission before O-bias is:

```math
\hat{E}_i^{final} = E_i^{(0)} + g_i \odot R_i
```

The newest change adds a learnable O-emission bias:

```math
E_{i,O}^{final} = \hat{E}_{i,O}^{final} + \beta_O
```

For non-O labels:

```math
E_{i,y}^{final} = \hat{E}_{i,y}^{final},\quad y \ne O
```

Here:

```math
\beta_O
```

is a trainable scalar parameter named `final_o_bias`.

The final prediction is:

```math
Y^{final} = CRFDecode(E^{final})
```

This means the model does not hard overwrite token labels. It learns how much
AU/document reasoning should modify token emissions.

## 4. Current Loss Function

The current total loss is:

```math
\mathcal{L}
= \mathcal{L}_{BIO}
+ \mathcal{L}_{AU}
+ \mathcal{L}_D
```

The BIO part is now a combined objective:

```math
\mathcal{L}_{BIO}
= \mathcal{L}_{FinalCRF}
+ \lambda_{init}\mathcal{L}_{InitialCRF}
+ \lambda_{off}\mathcal{L}_{OfficialToken}
```

Current default weights:

```text
lambda_init = 0.3
lambda_off  = 0.5
```

The final CRF loss is:

```math
\mathcal{L}_{FinalCRF}
= -\log P(Y^{gold}_{BIO} \mid E^{final})
```

The initial CRF auxiliary loss is:

```math
\mathcal{L}_{InitialCRF}
= -\log P(Y^{gold}_{BIO} \mid E^{(0)})
```

The official token auxiliary loss collapses 5-class emissions into 3 official
token classes:

```math
z_{non} = E_O
```

```math
z_{con} = logsumexp(E_{B-Con}, E_{I-Con})
```

```math
z_{pro} = logsumexp(E_{B-Pro}, E_{I-Pro})
```

Then:

```math
\mathcal{L}_{OfficialToken}
= CE([z_{non}, z_{con}, z_{pro}], y^{official})
```

The complete training objective is therefore:

```math
\mathcal{L}
=
\mathcal{L}_{FinalCRF}
+ 0.3\mathcal{L}_{InitialCRF}
+ 0.5\mathcal{L}_{OfficialToken}
+ \mathcal{L}_{AU}
+ \mathcal{L}_D
```

## 5. Latest O-Bias Change

Before the latest code change, an offline O-emission bias calibration was added
in:

```text
calibrate_o_emission_bias.py
```

It scanned a scalar beta added to the O emission after training. On the previous
completed run, beta 0.325 improved:

```text
Dev Official Token F1:  0.689888 -> 0.731465
Test Official Token F1: 0.687531 -> 0.702072
```

The newest code change moves this idea into the model itself:

```text
src/models.py
```

The model now has a learnable scalar:

```text
final_o_bias
```

The batch experiment launcher initializes it to:

```text
initial_o_bias = 0.325
```

This is intentionally initialized from the best offline calibration value. It
can still be updated by gradient descent during training.

When running `run_AURC_token.py` directly, the default is:

```text
--initial_o_bias 0.0
```

When running `run_batch_size_experiments.py`, the default is:

```text
--initial_o_bias 0.325
```

This makes the batch launcher use the current best practical setting while
keeping the lower-level training script suitable for ablation.

## 6. Current Verified Code Changes

Important source locations:

```text
src/models.py
  TokenBERT
  final_o_bias
  Final emission correction
  Final/Initial/OfficialToken losses

src/run_AURC_token.py
  Dev Official Token macro-F1 checkpoint selection
  --initial_crf_loss_weight
  --official_token_loss_weight
  --initial_o_bias
  O-bias logging in metrics

run_batch_size_experiments.py
  Sequential batch-size experiment runner
  Default batch sizes: 32, 64
  Default initial_o_bias: 0.325

calibrate_o_emission_bias.py
  Offline no-retraining O-bias calibration
```

Current validation:

```text
python -m unittest discover -s tests -p "test_*.py" -v
24 tests passed
```

Compile check:

```text
python -m compileall src run_batch_size_experiments.py calibrate_o_emission_bias.py
passed
```

## 7. Latest Observed Experiment Results

### 7.1 Latest run from pasted console log

The latest pasted run used:

```text
batch_size = 32
hetgat_heads = 1
initial_crf_loss_weight = 0.3
official_token_loss_weight = 0.5
early stopping = Dev Official Token macro-F1
```

Best epoch:

```text
epoch 10
```

Best Dev:

```text
Official Token F1: 0.6896
Official Segment F1: 0.6474
Official Sentence F1: 0.6888
Final 5-class BIO F1: 0.6144
AU span F1: 0.5090
AU stance Acc/F1: 0.8400 / 0.8397
Entity F1: 0.4348
Document F1: 0.6767
```

Best Test:

```text
Official Token F1: 0.6988
Official Segment F1: 0.6586
Official Sentence F1: 0.7199
Initial/Final 5-class BIO F1: 0.6114 / 0.6144
Feedback delta: +0.0030
AU span F1: 0.4548
AU stance Acc/F1: 0.8824 / 0.8821
Entity F1: 0.4023
Document F1: 0.7114
```

Interpretation:

```text
The OfficialTokenCE auxiliary loss and Official Token F1 checkpoint selection
are working, but the final feedback module is still only slightly improving
Initial BIO on the best test checkpoint.
```

The feedback gain:

```text
Initial BIO F1 -> Final BIO F1: 0.6114 -> 0.6144
delta = +0.0030
```

This means the graph/document feedback is useful, but not yet strongly enough.

### 7.2 Previous offline calibration result

The previous no-retraining O-bias calibration achieved:

```text
Test Official Token F1: 0.702072
```

That is why the latest code now makes the O-bias trainable.

## 8. Current Bottleneck Diagnosis

The model already predicts AU stance quite well:

```text
Test AU stance F1 around 0.88
```

The current bottleneck is not mainly AU stance classification. The bottleneck is
how well graph-level stance reasoning is converted back into better token-level
official labels.

The most important current observation:

```text
Final BIO F1 only slightly improves Initial BIO F1.
```

Therefore the next improvements should target:

```text
1. Better O/non calibration.
2. More stable feedback gating.
3. Better alignment between BIO loss and Official Token F1.
4. Avoiding excessive non-token false positives.
```

## 9. Recommended Next Experiment

Run only batch size 32 first. Do not immediately run batch size 64, because the
goal is to isolate whether trainable O-bias improves the official token metric.

Recommended command:

```powershell
D:\anaconda\envs\aurc\python.exe F:\AURC-master\run_batch_size_experiments.py --batch_sizes 32
```

If the actual project path is:

```text
C:\Users\TYUST\Desktop\AURC-master
```

then use:

```powershell
D:\anaconda\envs\aurc\python.exe C:\Users\TYUST\Desktop\AURC-master\run_batch_size_experiments.py --batch_sizes 32
```

The expected startup log should include:

```text
Best checkpoint criterion = Dev Official Token macro-F1
Loss = BIOCombined + AU stance + Document stance;
BIOCombined = FinalCRF + 0.300*InitialCRF + 0.500*OfficialTokenCE
Final O-emission bias: learnable scalar initialized to 0.325
```

During training, each split log should include:

```text
O-bias=...
```

After training, compare:

```text
Test Official Token F1
Test Final 5-class BIO F1
Initial/Final BIO delta
final_o_bias
```

Target:

```text
Test Official Token F1 > 0.7021
```

## 10. Suggested Follow-Up Steps

If trainable O-bias improves the score:

```text
Keep initial_o_bias=0.325 and run another seed for stability.
Then test batch_size=64 only after batch_size=32 is stable.
```

If trainable O-bias does not improve the score:

```text
1. Try initial_o_bias=0.25 and 0.40.
2. Add a feedback gate regularizer so the gate does not over-edit confident tokens.
3. Add an O-vs-argument binary auxiliary loss before Pro/Con polarity.
4. Tune official_token_loss_weight from 0.5 to 0.8 or 1.0.
```

If AU stance remains high but token F1 is flat:

```text
Focus on token feedback design, not HetGAT capacity.
Do not increase HetGAT heads yet.
```

If AU span F1 drops while official token F1 rises:

```text
This means the model is improving coarse official token classification but
hurting exact AU boundary recovery. Whether this is acceptable depends on the
paper's primary metric. The current primary checkpoint metric is Official Token
macro-F1, so this tradeoff is currently allowed.
```

## 11. Important Notes for Paper Writing

A clean description of the current method is:

```text
We propose a stance-guided token feedback framework. The model first obtains
syntax-enhanced token representations through dependency-aware semantic-syntax
fusion, predicts an initial BIO sequence with CRF, constructs AU and document
nodes, performs batch-level heterogeneous graph reasoning, predicts AU/document
stance, and maps high-level stance evidence back to token emissions through a
gated feedback module before final CRF decoding.
```

The key novelty is not just predicting AU stance and document stance. The key
novelty is:

```text
AU/document-level reasoning is fed back into token-level BIO prediction.
```

The central expected empirical claim is:

```text
Final BIO / Official Token F1 should be higher than Initial BIO / Official
Token F1 because graph-level stance reasoning corrects token-level polarity
and non/argument confusion.
```

The current empirical situation is promising but not finished:

```text
AU stance is strong, document stance is reasonable, and final BIO sometimes
improves initial BIO. The next experiments should strengthen and calibrate the
feedback path.
```

