#!/usr/bin/env python
"""Pure-PyTorch batch-level AU/Document heterogeneous graph attention."""

from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


RELATION_TYPES = (
    "self",
    "au_to_au",
    "au_to_doc",
    "doc_to_au",
    "doc_to_doc",
)


def _empty_edges(device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    return {
        "edge_index": torch.empty((2, 0), dtype=torch.long, device=device),
        "edge_weight": torch.empty((0,), dtype=dtype, device=device),
    }


def build_batch_heterogeneous_edges(
    au_representations: torch.Tensor,
    au_records: Sequence[Dict[str, object]],
    document_count: int,
    topics: Sequence[str],
    dependency_adj_wordpiece: torch.Tensor,
    semantic_threshold: float,
    top_k: int,
    syntax_hops: int = 1,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, object]]:
    """Build one graph for a complete mini-batch, without Topic nodes.

    Node order is ``[all AU nodes, one Document node per sample]``. AU semantic
    and syntax evidence are summed into one ``au_to_au`` relation rather than
    exposed as separate HetGAT relation types. Same-document AU pairs retain a
    structural base edge; semantic evidence may also connect AUs across samples.
    Documents connect when their raw topic strings are equal.
    """
    device = au_representations.device
    dtype = au_representations.dtype
    au_count = int(au_representations.shape[0])
    document_count = int(document_count)
    if len(au_records) != au_count:
        raise ValueError("au_records length must match au_representations")
    if len(topics) != document_count:
        raise ValueError("topics length must match document_count")
    node_count = au_count + document_count
    edges = {relation: _empty_edges(device, dtype) for relation in RELATION_TYPES}

    edge_lists: Dict[str, List[Tuple[int, int, torch.Tensor]]] = {
        relation: [] for relation in RELATION_TYPES
    }
    one = au_representations.new_tensor(1.0)

    for node_index in range(node_count):
        edge_lists["self"].append((node_index, node_index, one))

    for au_index, record in enumerate(au_records):
        sample_index = int(record["sample_index"])
        document_index = au_count + sample_index
        edge_lists["au_to_doc"].append((au_index, document_index, one))
        edge_lists["doc_to_au"].append((document_index, au_index, one))

    for source in range(document_count):
        for target in range(document_count):
            if source != target and str(topics[source]) == str(topics[target]):
                edge_lists["doc_to_doc"].append(
                    (au_count + source, au_count + target, one)
                )

    if au_count:
        normalized_au = F.normalize(au_representations, p=2, dim=-1, eps=1e-8)
        semantic_similarity = normalized_au @ normalized_au.transpose(0, 1)
    else:
        semantic_similarity = au_representations.new_zeros((0, 0))

    semantic_selected = set()
    for source in range(au_count):
        candidates = [
            target
            for target in range(au_count)
            if target != source
            and float(semantic_similarity[source, target].detach().cpu())
            >= float(semantic_threshold)
        ]
        candidates.sort(
            key=lambda target: float(
                semantic_similarity[source, target].detach().cpu()
            ),
            reverse=True,
        )
        if int(top_k) > 0:
            candidates = candidates[: int(top_k)]
        for target in candidates:
            semantic_selected.add((source, target))
    syntax_reach_by_sample = []
    for sample_index in range(document_count):
        syntax_reach = dependency_adj_wordpiece[sample_index] > 0
        for _ in range(max(1, int(syntax_hops)) - 1):
            syntax_reach = (
                syntax_reach.to(dtype=torch.float32)
                @ (dependency_adj_wordpiece[sample_index] > 0).to(dtype=torch.float32)
            ) > 0
        syntax_reach_by_sample.append(syntax_reach)

    syntax_selected = set()
    for source in range(au_count):
        source_record = au_records[source]
        source_sample = int(source_record["sample_index"])
        source_span = source_record["span"]
        valid_token_count = int(dependency_adj_wordpiece.shape[-1])
        source_start = max(0, min(int(source_span["start"]), valid_token_count))
        source_end = max(source_start, min(int(source_span["end"]), valid_token_count))
        for target in range(au_count):
            if source == target:
                continue
            target_record = au_records[target]
            target_sample = int(target_record["sample_index"])
            if source_sample != target_sample:
                continue
            target_span = target_record["span"]
            target_start = max(0, min(int(target_span["start"]), valid_token_count))
            target_end = max(target_start, min(int(target_span["end"]), valid_token_count))
            related = (
                source_end > source_start
                and target_end > target_start
                and bool(
                    syntax_reach_by_sample[source_sample][
                        source_start:source_end, target_start:target_end
                    ].any().item()
                )
            )
            if related:
                syntax_selected.add((source, target))

    debug_edges = []
    for source in range(au_count):
        source_sample = int(au_records[source]["sample_index"])
        for target in range(au_count):
            if source == target:
                continue
            target_sample = int(au_records[target]["sample_index"])
            same_document = source_sample == target_sample
            semantic_edge = (source, target) in semantic_selected
            syntax_edge = (source, target) in syntax_selected
            if not same_document and not semantic_edge and not syntax_edge:
                continue
            weight = au_representations.new_tensor(1.0 if same_document else 0.0)
            semantic_value = semantic_similarity[source, target]
            if semantic_edge:
                weight = weight + semantic_value.clamp_min(1e-6)
            if syntax_edge:
                weight = weight + 1.0
            edge_lists["au_to_au"].append((source, target, weight.clamp_min(1e-6)))
            debug_edges.append(
                {
                    "source": source,
                    "target": target,
                    "source_sample": source_sample,
                    "target_sample": target_sample,
                    "same_document": same_document,
                    "semantic_similarity": float(semantic_value.detach().cpu()),
                    "semantic_edge": semantic_edge,
                    "syntax_edge": syntax_edge,
                    "combined_weight": float(weight.detach().cpu()),
                }
            )

    for relation, relation_edges in edge_lists.items():
        if not relation_edges:
            continue
        sources = torch.tensor(
            [source for source, _, _ in relation_edges],
            dtype=torch.long,
            device=device,
        )
        targets = torch.tensor(
            [target for _, target, _ in relation_edges],
            dtype=torch.long,
            device=device,
        )
        weights = torch.stack(
            [
                weight if torch.is_tensor(weight) else au_representations.new_tensor(weight)
                for _, _, weight in relation_edges
            ]
        ).to(dtype=dtype)
        edges[relation] = {
            "edge_index": torch.stack([sources, targets], dim=0),
            "edge_weight": weights,
        }

    return edges, {
        "node_order": "all_aus_then_documents",
        "au_count": au_count,
        "document_count": document_count,
        "au_au_edges": debug_edges,
        "document_topic_edges": sum(
            1 for _, _, _ in edge_lists["doc_to_doc"]
        ),
    }


class RelationAwareGATLayer(nn.Module):
    """A relation-specific, multi-head graph-attention layer."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        relation_types: Sequence[str] = RELATION_TYPES,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size={} must be divisible by hetgat_heads={}"
                .format(hidden_size, num_heads)
            )
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_size = self.hidden_size // self.num_heads
        self.relation_types = tuple(relation_types)
        self.transforms = nn.ModuleDict(
            {
                relation: nn.Linear(hidden_size, hidden_size, bias=False)
                for relation in self.relation_types
            }
        )
        self.attention_source = nn.ParameterDict(
            {
                relation: nn.Parameter(torch.empty(num_heads, self.head_size))
                for relation in self.relation_types
            }
        )
        self.attention_target = nn.ParameterDict(
            {
                relation: nn.Parameter(torch.empty(num_heads, self.head_size))
                for relation in self.relation_types
            }
        )
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in self.attention_source.values():
            nn.init.xavier_uniform_(parameter)
        for parameter in self.attention_target.values():
            nn.init.xavier_uniform_(parameter)

    def forward(
        self,
        node_states: torch.Tensor,
        edges: Dict[str, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        node_count = int(node_states.shape[0])
        aggregate = node_states.new_zeros(
            (node_count, self.num_heads, self.head_size)
        )
        for relation in self.relation_types:
            relation_edges = edges.get(relation)
            if relation_edges is None or relation_edges["edge_index"].numel() == 0:
                continue
            edge_index = relation_edges["edge_index"]
            sources, targets = edge_index[0], edge_index[1]
            transformed = self.transforms[relation](node_states).view(
                node_count, self.num_heads, self.head_size
            )
            source_states = transformed.index_select(0, sources)
            target_states = transformed.index_select(0, targets)
            logits = (
                source_states * self.attention_source[relation].unsqueeze(0)
            ).sum(dim=-1)
            logits = logits + (
                target_states * self.attention_target[relation].unsqueeze(0)
            ).sum(dim=-1)
            logits = F.leaky_relu(logits, negative_slope=0.2)
            logits = logits + torch.log(
                relation_edges["edge_weight"].clamp_min(1e-8)
            ).unsqueeze(-1)

            attention = torch.zeros_like(logits)
            for target in torch.unique(targets):
                target_mask = targets == target
                attention[target_mask] = torch.softmax(logits[target_mask], dim=0)
            attention = self.dropout(attention)
            messages = source_states * attention.unsqueeze(-1)
            aggregate.index_add_(0, targets, messages)

        output = aggregate.reshape(node_count, self.hidden_size)
        output = self.output_projection(output)
        output = self.dropout(F.elu(output))
        return self.layer_norm(node_states + output)


class HeterogeneousGAT(nn.Module):
    """Stacked relation-aware GAT producing batch AU and Document states."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 1,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                RelationAwareGATLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        node_states: torch.Tensor,
        edges: Dict[str, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        for layer in self.layers:
            node_states = layer(node_states, edges)
        return node_states
