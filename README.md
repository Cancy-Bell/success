# AURC

Accompanying repository of the AAAI-20 paper [Fine-Grained Argument Unit Recognition and Classification](https://doi.org/10.1609/aaai.v34i05.6438).

> The dataset was updated with cleaner parsing and encoding. Sentence and label counts remain unchanged.

### Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Data download

```bash
sh download.sh
```

### Annotations

`merged_segments` contains gold labels aggregated from five annotators. Each
entry records whether no argument was found, character spans, and their `con`
or `pro` stance labels.

### Training and inference

```bash
python src/run_AURC_token.py \
  --train \
  --crf \
  --target_domain In-Domain \
  --input_file AURC_DATA_dict.json \
  --data_dir ./data \
  --output_dir ./models/aurc_graph \
  --pretrained_weights bert-large-cased-whole-word-masking \
  --epochs 30 \
  --max_sequence_length 64 \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --test_batch_size 32 \
  --gcn_layers 2 \
  --hetgat_layers 2 \
  --hetgat_heads 1 \
  --au_semantic_threshold 0.5 \
  --au_top_k 3 \
  --au_syntax_hops 1 \
  --early_stop_patience 5 \
  --early_stop_min_delta 0.0 \
  --save_predictions
```

The complete model is trained end-to-end from epoch 1. A joint Topic+Sentence
BERT supplies sentence WordPiece semantics, while a separate Topic-only BERT
supplies Topic CLS. Dependency GCN states and semantic states enter gated
Semantic-Syntax Fusion without an additional BERT residual or Token-Topic
Fusion.

Gold BIO spans construct training AU nodes; initial CRF predictions construct
evaluation AU nodes. Topic-conditioned masked attention pools AU
representations, while full attention pools Document representations. One
mini-batch forms one AU/Document HetGAT graph. It has no Topic node, combines
semantic and syntax evidence in one AU relation, connects AU/Document nodes
within samples, and connects Documents that share a topic. The default HetGAT
head count is one.

AU and Document stance probabilities are mapped back to token correction
features. A learned gate adds the correction to initial BIO emissions, and a
final CRF produces the final five-class BIO sequence. Every epoch optimizes the
unweighted objective `L_FinalBIO + L_AU + L_Document`. Early stopping and best
checkpoint selection use Dev final five-class BIO macro-F1.

Every prediction JSONL retains `gold_bio`, `initial_crf_bio`, `final_bio`,
initial/final token probabilities, AU stance probabilities, Document stance
probabilities, and token feedback gates. Official AURC metrics are calculated
from final BIO labels. Test results never participate in model selection.

### Sequential batch-size comparison

```bash
python run_batch_size_experiments.py
```

Each invocation creates a timestamped directory under
`models/batch_size_comparison`, containing independent `batch_size_32` and
`batch_size_64` artifacts plus `batch_size_comparison.json`.

### Citation

```bibtex
@inproceedings{trautmann2020fine,
  title = {Fine-Grained Argument Unit Recognition and Classification},
  author = {Dietrich Trautmann and Johannes Daxenberger and Christian Stab and Hinrich Schutze and Iryna Gurevych},
  booktitle = {The Thirty-Fourth AAAI Conference on Artificial Intelligence},
  publisher = {AAAI Press},
  year = {2020}
}
```
