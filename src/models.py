#!/usr/bin/env python
"""Official TokenBERT extended to the full end-to-end AURC graph model."""

from typing import Dict, List, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from transformers import BertConfig, BertForTokenClassification, BertModel

from graph_utils import HeterogeneousGAT, build_batch_heterogeneous_edges
from label_utils import (
    AU_STANCE_TO_ID,
    B_CON,
    B_PRO,
    O,
    I_CON,
    I_PRO,
    bio_to_spans,
    repair_bio_sequence,
)


class LinearChainCRF(nn.Module):
    """Batch-first linear-chain CRF with BIO transition constraints."""

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = int(num_tags)
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        transition_constraint = torch.zeros(num_tags, num_tags)
        start_constraint = torch.zeros(num_tags)
        if num_tags == 5:
            # transitions[previous_tag, next_tag]
            for previous in range(num_tags):
                if previous not in (B_PRO, I_PRO):
                    transition_constraint[previous, I_PRO] = -10000.0
                if previous not in (B_CON, I_CON):
                    transition_constraint[previous, I_CON] = -10000.0
            start_constraint[I_PRO] = -10000.0
            start_constraint[I_CON] = -10000.0
        self.register_buffer("transition_constraint", transition_constraint)
        self.register_buffer("start_constraint", start_constraint)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def _validate(self, emissions, tags=None, mask=None) -> torch.Tensor:
        if emissions.dim() != 3 or emissions.size(-1) != self.num_tags:
            raise ValueError("emissions must have shape [batch, length, num_tags]")
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
        else:
            mask = mask.bool()
        if mask.shape != emissions.shape[:2]:
            raise ValueError("CRF mask shape does not match emissions")
        if mask.shape[1] == 0 or not bool(mask[:, 0].all().item()):
            raise ValueError("every CRF sequence must contain a first sentence token")
        if tags is not None and tags.shape != emissions.shape[:2]:
            raise ValueError("CRF tag shape does not match emissions")
        return mask

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Return log-likelihood (callers negate it for NLL)."""
        mask = self._validate(emissions, tags=tags, mask=mask)
        transitions = self.transitions + self.transition_constraint
        start_transitions = self.start_transitions + self.start_constraint

        numerator = start_transitions[tags[:, 0]]
        numerator = numerator + emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        for timestep in range(1, emissions.size(1)):
            active = mask[:, timestep].to(dtype=emissions.dtype)
            transition_score = transitions[tags[:, timestep - 1], tags[:, timestep]]
            emission_score = emissions[:, timestep].gather(
                1, tags[:, timestep : timestep + 1]
            ).squeeze(1)
            numerator = numerator + (transition_score + emission_score) * active
        lengths = mask.long().sum(dim=1)
        last_tags = tags.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
        numerator = numerator + self.end_transitions[last_tags]

        score = start_transitions.unsqueeze(0) + emissions[:, 0]
        for timestep in range(1, emissions.size(1)):
            next_score = (
                score.unsqueeze(2)
                + transitions.unsqueeze(0)
                + emissions[:, timestep].unsqueeze(1)
            )
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[:, timestep].unsqueeze(1), next_score, score)
        denominator = torch.logsumexp(score + self.end_transitions.unsqueeze(0), dim=1)
        likelihood = numerator - denominator
        if reduction == "none":
            return likelihood
        if reduction == "sum":
            return likelihood.sum()
        if reduction == "mean":
            return likelihood.mean()
        raise ValueError("unknown CRF reduction: {}".format(reduction))

    def decode(
        self, emissions: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        mask = self._validate(emissions, mask=mask)
        transitions = self.transitions + self.transition_constraint
        start_transitions = self.start_transitions + self.start_constraint
        paths: List[List[int]] = []
        for sample_index in range(emissions.size(0)):
            length = int(mask[sample_index].long().sum().item())
            score = start_transitions + emissions[sample_index, 0]
            history: List[torch.Tensor] = []
            for timestep in range(1, length):
                next_score = score.unsqueeze(1) + transitions
                best_score, best_previous = next_score.max(dim=0)
                score = best_score + emissions[sample_index, timestep]
                history.append(best_previous)
            score = score + self.end_transitions
            best_last = int(score.argmax(dim=0).item())
            path = [best_last]
            for best_previous in reversed(history):
                best_last = int(best_previous[best_last].item())
                path.append(best_last)
            path.reverse()
            paths.append(path)
        return paths


class DependencyGCN(nn.Module):
    """Multi-layer, mask-safe dependency GCN over sentence WordPieces."""

    def __init__(self, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def normalize_adjacency(
        adjacency_wordpiece: torch.Tensor, sentence_mask: torch.Tensor
    ) -> torch.Tensor:
        """Separately compute D^-1/2 (A_wordpiece + I) D^-1/2."""
        valid_pairs = sentence_mask.unsqueeze(1) & sentence_mask.unsqueeze(2)
        adjacency = adjacency_wordpiece * valid_pairs.to(adjacency_wordpiece.dtype)
        batch_size, sentence_length, _ = adjacency.shape
        identity = torch.eye(
            sentence_length, dtype=adjacency.dtype, device=adjacency.device
        ).unsqueeze(0).expand(batch_size, -1, -1)
        adjacency_with_self_loops = adjacency + identity * sentence_mask.unsqueeze(
            -1
        ).to(adjacency.dtype)
        degree = adjacency_with_self_loops.sum(dim=-1).clamp_min(1e-8)
        inverse_sqrt_degree = degree.pow(-0.5) * sentence_mask.to(adjacency.dtype)
        adjacency_wordpiece_norm = (
            inverse_sqrt_degree.unsqueeze(-1)
            * adjacency_with_self_loops
            * inverse_sqrt_degree.unsqueeze(-2)
        )
        return adjacency_wordpiece_norm

    def forward(
        self,
        sentence_states: torch.Tensor,
        adjacency_wordpiece: torch.Tensor,
        sentence_mask: torch.Tensor,
    ) -> torch.Tensor:
        # sentence_states: [batch_size, max_sentence_wp_len, hidden_size]
        # adjacency_wordpiece: [batch_size, max_sentence_wp_len, max_sentence_wp_len]
        adjacency_wordpiece_norm = self.normalize_adjacency(
            adjacency_wordpiece=adjacency_wordpiece,
            sentence_mask=sentence_mask,
        )
        states = sentence_states
        for linear, layer_norm in zip(self.layers, self.layer_norms):
            messages = torch.bmm(adjacency_wordpiece_norm, states)
            update = self.dropout(F.gelu(linear(messages)))
            states = layer_norm(states + update)
            states = states * sentence_mask.unsqueeze(-1).to(states.dtype)
        return states


class PairFusion(nn.Module):
    """Gated fusion of two aligned representations without an outer residual."""

    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.projection = nn.Linear(4 * hidden_size, hidden_size)
        self.gate = nn.Linear(2 * hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [states, context, states * context, torch.abs(states - context)], dim=-1
        )
        candidate = self.dropout(F.gelu(self.projection(features)))
        gate = torch.sigmoid(self.gate(torch.cat([states, context], dim=-1)))
        return self.layer_norm(gate * candidate + (1.0 - gate) * states)


class TopicAttentionPooling(nn.Module):
    """Use Topic CLS as query for masked AU or full-document token attention."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = float(hidden_size) ** -0.5

    def forward(
        self,
        topic_repr: torch.Tensor,
        token_states: torch.Tensor,
        token_mask: torch.Tensor,
    ):
        token_mask = token_mask.bool()
        if not bool(token_mask.any().item()):
            raise ValueError("attention pooling requires at least one valid token")
        query = self.query(topic_repr)
        keys = self.key(token_states)
        values = self.value(token_states)
        scores = torch.matmul(keys, query) * self.scale
        scores = scores.masked_fill(~token_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=0)
        pooled = torch.sum(weights.unsqueeze(-1) * values, dim=0)
        return pooled, weights


class TokenBERT(nn.Module):
    """Dual-BERT, batch-HetGAT and stance-guided final BIO-CRF model."""

    def __init__(
        self,
        num_labels: int,
        model_name: str,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        batch_first: bool = True,
        use_crf: bool = True,
        config: Optional[BertConfig] = None,
        initialize_from_pretrained: bool = True,
        gcn_layers: int = 2,
        gcn_dropout: float = 0.1,
        hetgat_layers: int = 2,
        hetgat_heads: int = 1,
        hetgat_dropout: float = 0.1,
        au_semantic_threshold: float = 0.5,
        au_top_k: int = 3,
        au_syntax_hops: int = 1,
        num_document_labels: int = 3,
        initial_crf_loss_weight: float = 0.3,
        official_token_loss_weight: float = 0.5,
        initial_o_bias: float = 0.0,
        local_files_only: bool = False,
    ):
        super().__init__()
        if int(num_labels) != 5:
            raise ValueError("the end-to-end AURC model requires exactly 5 BIO labels")
        self.num_labels = int(num_labels)
        self.batch_first = bool(batch_first)
        self.use_crf = bool(use_crf)
        self.au_semantic_threshold = float(au_semantic_threshold)
        self.au_top_k = int(au_top_k)
        self.au_syntax_hops = int(au_syntax_hops)
        self.initial_crf_loss_weight = float(initial_crf_loss_weight)
        self.official_token_loss_weight = float(official_token_loss_weight)
        if self.initial_crf_loss_weight < 0.0:
            raise ValueError("initial_crf_loss_weight must be >= 0")
        if self.official_token_loss_weight < 0.0:
            raise ValueError("official_token_loss_weight must be >= 0")

        if config is None:
            config = BertConfig.from_pretrained(
                model_name,
                num_labels=self.num_labels,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
        else:
            config.num_labels = self.num_labels
            config.output_hidden_states = output_hidden_states
            config.output_attentions = output_attentions
        if initialize_from_pretrained:
            self.tokenbert = BertForTokenClassification.from_pretrained(
                model_name, config=config, local_files_only=local_files_only
            )
            self.topic_bert = BertModel.from_pretrained(
                model_name, config=config, local_files_only=local_files_only
            )
        else:
            self.tokenbert = BertForTokenClassification(config)
            self.topic_bert = BertModel(config)

        hidden_size = int(config.hidden_size)
        dropout = float(config.hidden_dropout_prob)
        self.dependency_gcn = DependencyGCN(
            hidden_size=hidden_size,
            num_layers=int(gcn_layers),
            dropout=float(gcn_dropout),
        )
        self.shared_dropout = nn.Dropout(dropout)
        self.semantic_syntax_fusion = PairFusion(hidden_size, dropout)
        self.crf = LinearChainCRF(self.num_labels) if self.use_crf else None

        self.au_attention_pool = TopicAttentionPooling(hidden_size)
        self.document_attention_pool = TopicAttentionPooling(hidden_size)
        self.hetgat = HeterogeneousGAT(
            hidden_size=hidden_size,
            num_heads=int(hetgat_heads),
            num_layers=int(hetgat_layers),
            dropout=float(hetgat_dropout),
        )
        self.au_stance_classifier = nn.Linear(2 * hidden_size, 2)
        self.document_classifier = nn.Linear(hidden_size, int(num_document_labels))
        self.document_to_token = nn.Linear(int(num_document_labels), self.num_labels)
        feedback_size = hidden_size + 2 * self.num_labels
        self.feedback_gate = nn.Linear(feedback_size, self.num_labels)
        self.feedback_correction = nn.Linear(feedback_size, self.num_labels)
        self.final_o_bias = nn.Parameter(
            torch.tensor(float(initial_o_bias), dtype=torch.float)
        )

    def _decode(self, emissions: torch.Tensor, sentence_mask: torch.Tensor):
        # CRF dynamic programming remains FP32 even when the surrounding model
        # uses CUDA autocast.
        emissions = emissions.float()
        if self.use_crf:
            return self.crf.decode(emissions, mask=sentence_mask)
        predictions = emissions.argmax(dim=-1)
        return [
            predictions[index, : int(mask.long().sum().item())].tolist()
            for index, mask in enumerate(sentence_mask)
        ]

    def _crf_loss(self, emissions, labels, sentence_mask):
        emissions = emissions.float()
        if labels is None:
            return emissions.sum() * 0.0
        if self.use_crf:
            return -self.crf(
                emissions, labels, mask=sentence_mask, reduction="mean"
            )
        return F.cross_entropy(emissions[sentence_mask], labels[sentence_mask])

    @staticmethod
    def _official_token_loss(emissions, labels, sentence_mask):
        """CE surrogate for official non/con/pro labels collapsed from BIO.

        The grouped logits preserve both B/I alternatives with log-sum-exp:
        non=O, con={B-Con,I-Con}, pro={B-Pro,I-Pro}.  This order matches the
        official AURC label order used by the metric implementation.
        """
        emissions = emissions.float()
        if labels is None:
            return emissions.sum() * 0.0
        official_logits = torch.stack(
            [
                emissions[..., 0],
                torch.logsumexp(emissions[..., [B_CON, I_CON]], dim=-1),
                torch.logsumexp(emissions[..., [B_PRO, I_PRO]], dim=-1),
            ],
            dim=-1,
        )
        target_map = torch.tensor(
            [0, 2, 2, 1, 1], dtype=torch.long, device=labels.device
        )
        official_targets = target_map[labels]
        mask = sentence_mask.bool()
        return F.cross_entropy(official_logits[mask], official_targets[mask])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        topic_input_ids: torch.Tensor,
        topic_attention_mask: torch.Tensor,
        topic_token_type_ids: torch.Tensor,
        sentence_indices: torch.Tensor,
        sentence_mask: torch.Tensor,
        dependency_adj_wordpiece: torch.Tensor,
        topics: Sequence[str],
        labels: Optional[torch.Tensor] = None,
        document_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        joint_outputs = self.tokenbert.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        topic_outputs = self.topic_bert(
            input_ids=topic_input_ids,
            attention_mask=topic_attention_mask,
            token_type_ids=topic_token_type_ids,
        )
        full_sequence_output = self.tokenbert.dropout(joint_outputs[0])
        # A dedicated BERT_T([CLS] Topic [SEP]) supplies the global Topic CLS.
        topic_repr = topic_outputs.last_hidden_state[:, 0]
        gather_indices = sentence_indices.unsqueeze(-1).expand(
            -1, -1, full_sequence_output.size(-1)
        )
        # bert_sentence_hidden: [batch_size, max_sentence_wp_len, hidden_size]
        bert_sentence_hidden = full_sequence_output.gather(1, gather_indices)
        bert_sentence_hidden = bert_sentence_hidden * sentence_mask.unsqueeze(-1).to(
            bert_sentence_hidden.dtype
        )

        syntax_hidden = self.dependency_gcn(
            sentence_states=bert_sentence_hidden,
            adjacency_wordpiece=dependency_adj_wordpiece,
            sentence_mask=sentence_mask.bool(),
        )
        semantic_syntax = self.semantic_syntax_fusion(
            bert_sentence_hidden, syntax_hidden
        )
        # No additional BERT residual and no independent Token-Topic fusion.
        token_hidden = self.shared_dropout(F.gelu(semantic_syntax))
        token_hidden = token_hidden * sentence_mask.unsqueeze(-1).to(token_hidden.dtype)

        initial_emissions = self.tokenbert.classifier(self.shared_dropout(token_hidden))
        initial_paths = self._decode(initial_emissions, sentence_mask.bool())
        initial_probs = torch.softmax(initial_emissions, dim=-1)

        batch_size = input_ids.size(0)
        if len(topics) != batch_size:
            raise ValueError("topics length must equal batch size")
        au_representations: List[torch.Tensor] = []
        au_records: List[Dict[str, object]] = []
        document_representations: List[torch.Tensor] = []
        document_attention_weights: List[torch.Tensor] = []
        for sample_index in range(batch_size):
            valid_length = int(sentence_mask[sample_index].long().sum().item())
            initial_path = repair_bio_sequence(
                [int(label) for label in initial_paths[sample_index]]
            )
            if self.training and labels is not None:
                graph_path = labels[sample_index, :valid_length].tolist()
                graph_source = "gold_bio"
            else:
                graph_path = initial_path
                graph_source = "predicted_bio"
            graph_spans = bio_to_spans(graph_path)
            sample_token_hidden = token_hidden[sample_index, :valid_length]
            for span in graph_spans:
                start, end = int(span["start"]), int(span["end"])
                if end <= start or start >= valid_length:
                    continue
                end = min(end, valid_length)
                au_mask = torch.zeros(
                    valid_length, dtype=torch.bool, device=input_ids.device
                )
                au_mask[start:end] = True
                pooled_au, attention_weights = self.au_attention_pool(
                    topic_repr[sample_index], sample_token_hidden, au_mask
                )
                au_records.append(
                    {
                        "sample_index": sample_index,
                        "span": {**span, "end": end},
                        "attention_weights": attention_weights,
                        "graph_source": graph_source,
                    }
                )
                au_representations.append(pooled_au)

            document_mask = torch.ones(
                valid_length, dtype=torch.bool, device=input_ids.device
            )
            document_repr, document_weights = self.document_attention_pool(
                topic_repr[sample_index], sample_token_hidden, document_mask
            )
            document_representations.append(document_repr)
            document_attention_weights.append(document_weights)

        if au_representations:
            all_au_repr = torch.stack(au_representations, dim=0)
        else:
            all_au_repr = token_hidden.new_empty((0, token_hidden.size(-1)))
        all_document_repr = torch.stack(document_representations, dim=0)
        node_states = torch.cat([all_au_repr, all_document_repr], dim=0)
        typed_edges, graph_debug = build_batch_heterogeneous_edges(
            au_representations=all_au_repr,
            au_records=au_records,
            document_count=batch_size,
            topics=topics,
            dependency_adj_wordpiece=dependency_adj_wordpiece,
            semantic_threshold=self.au_semantic_threshold,
            top_k=self.au_top_k,
            syntax_hops=self.au_syntax_hops,
        )
        graph_states = self.hetgat(node_states=node_states, edges=typed_edges)
        au_count = len(au_records)
        graph_au_repr = graph_states[:au_count]
        graph_document_repr = graph_states[au_count:]

        if au_count:
            au_topic_repr = torch.stack(
                [topic_repr[int(record["sample_index"])] for record in au_records], dim=0
            )
            au_logits = self.au_stance_classifier(
                torch.cat([graph_au_repr, au_topic_repr], dim=-1)
            )
            au_probs = torch.softmax(au_logits, dim=-1)
        else:
            au_logits = token_hidden.new_empty((0, 2))
            au_probs = token_hidden.new_empty((0, 2))
        document_logits = self.document_classifier(graph_document_repr)
        document_probs = torch.softmax(document_logits, dim=-1)

        zero = initial_emissions.sum() * 0.0
        au_loss_logits: List[torch.Tensor] = []
        au_loss_targets: List[int] = []
        if labels is not None:
            gold_spans_by_sample = [
                bio_to_spans(
                    labels[index, : int(sentence_mask[index].long().sum().item())].tolist()
                )
                for index in range(batch_size)
            ]
            gold_maps = [
                {
                    (int(span["start"]), int(span["end"])): AU_STANCE_TO_ID[
                        str(span["stance"]).lower()
                    ]
                    for span in spans
                }
                for spans in gold_spans_by_sample
            ]
            for au_index, record in enumerate(au_records):
                sample_index = int(record["sample_index"])
                span = record["span"]
                span_key = (int(span["start"]), int(span["end"]))
                if span_key in gold_maps[sample_index]:
                    au_loss_logits.append(au_logits[au_index])
                    au_loss_targets.append(gold_maps[sample_index][span_key])
        if au_loss_logits:
            au_targets = torch.tensor(
                au_loss_targets, dtype=torch.long, device=input_ids.device
            )
            au_loss = F.cross_entropy(torch.stack(au_loss_logits), au_targets)
        else:
            au_loss = zero
        document_loss = (
            F.cross_entropy(document_logits, document_labels)
            if document_labels is not None
            else zero
        )

        au_token_correction = token_hidden.new_zeros(
            (batch_size, token_hidden.size(1), self.num_labels)
        )
        for au_index, record in enumerate(au_records):
            sample_index = int(record["sample_index"])
            span = record["span"]
            start, end = int(span["start"]), int(span["end"])
            pro_probability, con_probability = au_probs[au_index]
            stance_correction = torch.stack(
                [
                    pro_probability * 0.0,
                    pro_probability,
                    pro_probability,
                    con_probability,
                    con_probability,
                ]
            )
            au_token_correction[sample_index, start:end] = stance_correction
        document_token_correction = self.document_to_token(document_probs)
        document_token_correction = document_token_correction.unsqueeze(1).expand(
            -1, token_hidden.size(1), -1
        )
        feedback_features = torch.cat(
            [token_hidden, au_token_correction, document_token_correction], dim=-1
        )
        feedback_gates = torch.sigmoid(self.feedback_gate(feedback_features))
        reasoning_correction = self.feedback_correction(feedback_features)
        final_emissions = initial_emissions + feedback_gates * reasoning_correction
        final_emissions = final_emissions.clone()
        final_emissions[..., O] = final_emissions[..., O] + self.final_o_bias
        final_bio_loss = self._crf_loss(
            final_emissions, labels, sentence_mask.bool()
        )
        initial_bio_loss = self._crf_loss(
            initial_emissions, labels, sentence_mask.bool()
        )
        official_token_loss = self._official_token_loss(
            final_emissions, labels, sentence_mask.bool()
        )
        bio_loss = (
            final_bio_loss
            + self.initial_crf_loss_weight * initial_bio_loss
            + self.official_token_loss_weight * official_token_loss
        )
        final_paths = self._decode(final_emissions, sentence_mask.bool())
        final_probs = torch.softmax(final_emissions, dim=-1)
        total_loss = bio_loss + au_loss + document_loss

        sample_outputs: List[Dict[str, object]] = []
        for sample_index in range(batch_size):
            valid_length = int(sentence_mask[sample_index].long().sum().item())
            local_au_indices = [
                index
                for index, record in enumerate(au_records)
                if int(record["sample_index"]) == sample_index
            ]
            sample_outputs.append(
                {
                    "initial_bio_ids": repair_bio_sequence(initial_paths[sample_index]),
                    "final_bio_ids": repair_bio_sequence(final_paths[sample_index]),
                    "initial_bio_probs": initial_probs[
                        sample_index, :valid_length
                    ].detach(),
                    "final_bio_probs": final_probs[sample_index, :valid_length].detach(),
                    "feedback_gate": feedback_gates[
                        sample_index, :valid_length
                    ].detach(),
                    "au_spans": [au_records[index]["span"] for index in local_au_indices],
                    "au_stance_probs": au_probs[local_au_indices].detach(),
                    "au_attention_weights": [
                        au_records[index]["attention_weights"][:valid_length].detach()
                        for index in local_au_indices
                    ],
                    "document_attention_weights": document_attention_weights[
                        sample_index
                    ].detach(),
                    "document_probs": document_probs[sample_index].detach(),
                    "graph_source": (
                        "gold_bio" if self.training and labels is not None else "predicted_bio"
                    ),
                    "graph_debug": graph_debug,
                }
            )

        return {
            "loss": total_loss,
            "total_loss": total_loss,
            "bio_loss": bio_loss,
            "final_bio_loss": final_bio_loss,
            "initial_bio_loss": initial_bio_loss,
            "official_token_loss": official_token_loss,
            "au_loss": au_loss,
            "document_loss": document_loss,
            "initial_bio_emissions": initial_emissions,
            "final_bio_emissions": final_emissions,
            "final_o_bias": self.final_o_bias,
            "topic_repr": topic_repr,
            "token_hidden": token_hidden,
            "sample_outputs": sample_outputs,
        }
