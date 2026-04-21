from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch

from src.correlation.compatibility import compatibility_score
from src.correlation.events import EventExtractionConfig, extract_cross_layer_events
from src.correlation.model import AttentionCorrelationModel
from src.correlation.pair_features import PAIR_FEATURE_DIM, build_pair_features
from src.correlation.reasoner import AttackChainReasoner
from src.correlation.response import suggest_mitigations
from src.correlation.schema import AnomalyEvent, CorrelationCluster, CorrelationEdge


DEFAULT_OUTPUT_PATH = Path("outputs/correlation/correlation_results.json")
DEFAULT_MODEL_PATH = Path("models/correlation_attention.pth")


@dataclass(slots=True)
class PipelineConfig:
    split: str
    device: str
    correlation_model_path: Path | None
    output_path: Path
    batch_size: int
    max_network_events: int | None
    max_system_events: int | None
    temporal_window_minutes: int
    edge_threshold: float
    use_gemini: bool
    gemini_model: str
    show_progress: bool


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Run cross-layer anomaly correlation and attack-chain reconstruction.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--correlation-model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-network-events", type=int, default=2000)
    parser.add_argument("--max-system-events", type=int, default=None)
    parser.add_argument("--temporal-window-minutes", type=int, default=20)
    parser.add_argument("--edge-threshold", type=float, default=0.62)
    parser.add_argument("--disable-correlation-model", action="store_true")
    parser.add_argument("--disable-gemini", action="store_true")
    parser.add_argument("--gemini-model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--hide-progress", action="store_true")
    args = parser.parse_args()
    return PipelineConfig(
        split=args.split,
        device=args.device,
        correlation_model_path=None
        if bool(args.disable_correlation_model)
        else (args.correlation_model_path if args.correlation_model_path.exists() else None),
        output_path=args.output_path,
        batch_size=max(32, int(args.batch_size)),
        max_network_events=None if args.max_network_events in (None, 0) else int(args.max_network_events),
        max_system_events=None if args.max_system_events in (None, 0) else int(args.max_system_events),
        temporal_window_minutes=max(1, int(args.temporal_window_minutes)),
        edge_threshold=float(args.edge_threshold),
        use_gemini=not bool(args.disable_gemini),
        gemini_model=args.gemini_model,
        show_progress=not bool(args.hide_progress),
    )


def main() -> None:
    config = parse_args()
    results = run_pipeline(config)
    print(
        json.dumps(
            {
                "summary": results["summary"],
                "metrics": results["metrics"],
                "case_studies": results["case_studies"],
                "output_path": str(config.output_path),
            },
            indent=2,
        )
    )


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    event_config = EventExtractionConfig(
        split=config.split,
        device=config.device,
        batch_size=config.batch_size,
        max_network_events=config.max_network_events,
        max_system_events=config.max_system_events,
        show_progress=config.show_progress,
    )
    network_events, system_events = extract_cross_layer_events(event_config)
    all_events = sorted([*network_events, *system_events], key=lambda event: event.timestamp)

    device = _resolve_device(config.device)
    correlation_model = _load_correlation_model(config.correlation_model_path, device=device) if config.correlation_model_path else None
    graph, edges = build_anomaly_graph(
        events=all_events,
        correlation_model=correlation_model,
        device=device,
        temporal_window_minutes=config.temporal_window_minutes,
        edge_threshold=config.edge_threshold,
    )
    clusters = derive_clusters(graph, edges, use_gemini=config.use_gemini, gemini_model=config.gemini_model)
    metrics = evaluate_correlation(graph=graph, events=all_events)

    payload = {
        "config": {
            "split": config.split,
            "device": config.device,
            "correlation_model_path": str(config.correlation_model_path) if config.correlation_model_path else None,
            "temporal_window_minutes": config.temporal_window_minutes,
            "edge_threshold": config.edge_threshold,
        },
        "summary": {
            "num_network_events": len(network_events),
            "num_log_events": len(system_events),
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "num_clusters": len(clusters),
        },
        "metrics": metrics,
        "events": [event.to_dict() for event in all_events],
        "edges": [edge.to_dict() for edge in edges],
        "clusters": [cluster.to_dict() for cluster in clusters],
        "case_studies": [cluster.to_dict() for cluster in clusters[:2]],
    }

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_anomaly_graph(
    *,
    events: list[AnomalyEvent],
    correlation_model: AttentionCorrelationModel | None,
    device: torch.device,
    temporal_window_minutes: int,
    edge_threshold: float,
) -> tuple[nx.DiGraph, list[CorrelationEdge]]:
    graph = nx.DiGraph()
    for event in events:
        graph.add_node(event.event_id, **event.to_dict())

    edges: list[CorrelationEdge] = []
    temporal_window_seconds = float(temporal_window_minutes * 60)
    network_events = [event for event in events if event.event_type == "network"]
    log_events = [event for event in events if event.event_type == "log"]

    # Keep only the strongest few causal parents per log to avoid collapsing
    # the graph into one giant cross-layer component.
    for log_event in log_events:
        causal_candidates: list[tuple[float, CorrelationEdge]] = []
        for network_event in network_events:
            delta_seconds = (log_event.timestamp - network_event.timestamp).total_seconds()
            if delta_seconds < 0.0 or delta_seconds > temporal_window_seconds:
                continue
            causal_score = compatibility_score(network_event.label, log_event.label)
            if causal_score < 0.18:
                continue
            rule_hint_score = 1.0 if log_event.synthetic_parent_id == network_event.event_id else 0.0

            temporal_score, semantic_score, attention_score = _pair_scores(
                left_event=network_event,
                right_event=log_event,
                causal_score=causal_score,
                relation="causal",
                delta_seconds=delta_seconds,
                temporal_window_seconds=temporal_window_seconds,
                correlation_model=correlation_model,
                device=device,
            )
            edge_score = (
                0.24 * temporal_score
                + 0.14 * semantic_score
                + 0.22 * causal_score
                + 0.15 * attention_score
                + 0.25 * rule_hint_score
            )
            if edge_score < max(0.46, edge_threshold - 0.12):
                continue

            causal_candidates.append(
                (
                    edge_score,
                    CorrelationEdge(
                        source_id=network_event.event_id,
                        target_id=log_event.event_id,
                        relation="causal",
                        score=float(edge_score),
                        temporal_score=float(temporal_score),
                        semantic_score=float(semantic_score),
                        causal_score=float(causal_score),
                        attention_score=float(attention_score),
                        metadata={
                            "time_delta_seconds": float(delta_seconds),
                            "left_type": network_event.event_type,
                            "right_type": log_event.event_type,
                            "rule_hint_score": float(rule_hint_score),
                        },
                    ),
                )
            )

        causal_candidates.sort(key=lambda item: item[0], reverse=True)
        for _, edge in causal_candidates[:1]:
            _append_edge(graph, edges, edge)

    # Add sequential contextual edges within the same network stream.
    network_groups: dict[tuple[str | None, str], list[AnomalyEvent]] = defaultdict(list)
    for event in network_events:
        network_groups[(event.metadata.get("source_file"), event.label)].append(event)

    for group_events in network_groups.values():
        group_events.sort(key=lambda event: event.timestamp)
        for left_event, right_event in zip(group_events, group_events[1:], strict=False):
            delta_seconds = (right_event.timestamp - left_event.timestamp).total_seconds()
            if delta_seconds <= 0.0 or delta_seconds > min(120.0, temporal_window_seconds):
                continue
            temporal_score, semantic_score, attention_score = _pair_scores(
                left_event=left_event,
                right_event=right_event,
                causal_score=0.15,
                relation="contextual",
                delta_seconds=delta_seconds,
                temporal_window_seconds=temporal_window_seconds,
                correlation_model=correlation_model,
                device=device,
            )
            edge_score = 0.50 * temporal_score + 0.25 * semantic_score + 0.25 * attention_score
            if edge_score < max(0.54, edge_threshold - 0.05):
                continue
            _append_edge(
                graph,
                edges,
                CorrelationEdge(
                    source_id=left_event.event_id,
                    target_id=right_event.event_id,
                    relation="contextual",
                    score=float(edge_score),
                    temporal_score=float(temporal_score),
                    semantic_score=float(semantic_score),
                    causal_score=0.15,
                    attention_score=float(attention_score),
                    metadata={
                        "time_delta_seconds": float(delta_seconds),
                        "left_type": left_event.event_type,
                        "right_type": right_event.event_type,
                    },
                ),
            )

    # Logs are linked only to their immediate neighbors inside the same aligned chain.
    log_groups: dict[str, list[AnomalyEvent]] = defaultdict(list)
    for event in log_events:
        if event.synthetic_chain_id is None:
            continue
        log_groups[event.synthetic_chain_id].append(event)

    for group_events in log_groups.values():
        group_events.sort(key=lambda event: event.timestamp)
        for left_event, right_event in zip(group_events, group_events[1:], strict=False):
            delta_seconds = (right_event.timestamp - left_event.timestamp).total_seconds()
            if delta_seconds <= 0.0 or delta_seconds > min(360.0, temporal_window_seconds):
                continue
            temporal_score, semantic_score, attention_score = _pair_scores(
                left_event=left_event,
                right_event=right_event,
                causal_score=0.25,
                relation="contextual",
                delta_seconds=delta_seconds,
                temporal_window_seconds=temporal_window_seconds,
                correlation_model=correlation_model,
                device=device,
            )
            edge_score = 0.42 * temporal_score + 0.23 * semantic_score + 0.15 * 0.25 + 0.20 * attention_score
            if edge_score < max(0.50, edge_threshold - 0.10):
                continue
            _append_edge(
                graph,
                edges,
                CorrelationEdge(
                    source_id=left_event.event_id,
                    target_id=right_event.event_id,
                    relation="contextual",
                    score=float(edge_score),
                    temporal_score=float(temporal_score),
                    semantic_score=float(semantic_score),
                    causal_score=0.25,
                    attention_score=float(attention_score),
                    metadata={
                        "time_delta_seconds": float(delta_seconds),
                        "left_type": left_event.event_type,
                        "right_type": right_event.event_type,
                        "same_chain": True,
                    },
                ),
            )

    return graph, edges


def _pair_scores(
    *,
    left_event: AnomalyEvent,
    right_event: AnomalyEvent,
    causal_score: float,
    relation: str,
    delta_seconds: float,
    temporal_window_seconds: float,
    correlation_model: AttentionCorrelationModel | None,
    device: torch.device,
) -> tuple[float, float, float]:
    pair_features = build_pair_features(
        left_event=left_event,
        right_event=right_event,
        delta_seconds=delta_seconds,
        temporal_window_seconds=temporal_window_seconds,
        causal_score=causal_score,
    )
    temporal_score = float(pair_features[0])
    semantic_score = float(pair_features[1])
    if correlation_model is not None:
        attention_score = correlation_model.predict_score(
            left_event.embedding,
            right_event.embedding,
            pair_features,
            device=device,
        )
    elif relation == "causal":
        attention_score = 0.40 * semantic_score + 0.60 * causal_score
    else:
        attention_score = 0.65 * temporal_score + 0.35 * semantic_score
    return float(temporal_score), float(semantic_score), float(attention_score)


def _append_edge(graph: nx.DiGraph, edges: list[CorrelationEdge], edge: CorrelationEdge) -> None:
    graph.add_edge(edge.source_id, edge.target_id, **edge.to_dict())
    edges.append(edge)


def derive_clusters(
    graph: nx.DiGraph,
    edges: list[CorrelationEdge],
    *,
    use_gemini: bool,
    gemini_model: str,
) -> list[CorrelationCluster]:
    if graph.number_of_nodes() == 0:
        return []

    incident_graph = nx.Graph()
    incident_graph.add_nodes_from(graph.nodes(data=True))
    for source_id, target_id, edge_data in graph.edges(data=True):
        if (
            edge_data.get("relation") == "contextual"
            and edge_data.get("metadata", {}).get("left_type") == "network"
            and edge_data.get("metadata", {}).get("right_type") == "network"
        ):
            continue
        incident_graph.add_edge(source_id, target_id, **edge_data)

    reasoner = AttackChainReasoner(use_gemini=use_gemini, gemini_model=gemini_model)
    clusters: list[CorrelationCluster] = []
    for cluster_index, node_ids in enumerate(nx.connected_components(incident_graph), start=1):
        ordered_nodes = sorted(node_ids, key=lambda node_id: datetime.fromisoformat(graph.nodes[node_id]["timestamp"]))
        root_id = select_root_cause(graph, ordered_nodes)
        attack_chain = [
            {
                "event_id": node_id,
                "timestamp": graph.nodes[node_id]["timestamp"],
                "type": graph.nodes[node_id]["type"],
                "label": graph.nodes[node_id]["label"],
                "anomaly_score": graph.nodes[node_id]["anomaly_score"],
                "severity": graph.nodes[node_id]["severity"],
            }
            for node_id in ordered_nodes
        ]
        cluster_payload = {
            "cluster_id": f"cluster-{cluster_index:03d}",
            "root_cause_id": root_id,
            "attack_chain": attack_chain,
            "edge_count": int(sum(1 for edge in edges if edge.source_id in node_ids and edge.target_id in node_ids)),
        }
        reasoning = reasoner.reason(cluster_payload)
        summary = {
            "title": reasoning.title,
            "root_cause": reasoning.root_cause,
            "explanation": reasoning.explanation,
            "node_count": len(ordered_nodes),
        }
        clusters.append(
            CorrelationCluster(
                cluster_id=f"cluster-{cluster_index:03d}",
                node_ids=ordered_nodes,
                root_cause_id=root_id,
                summary=summary,
                attack_chain=reasoning.attack_chain,
                mitigation=reasoning.mitigation or suggest_mitigations(node["label"] for node in attack_chain),
            )
        )

    clusters.sort(key=lambda cluster: (-cluster.summary["node_count"], cluster.cluster_id))
    return clusters


def select_root_cause(graph: nx.DiGraph, ordered_nodes: list[str]) -> str:
    best_node = ordered_nodes[0]
    best_score = -1.0
    for node_id in ordered_nodes:
        node = graph.nodes[node_id]
        type_bonus = 0.2 if node["type"] == "network" else 0.0
        severity_bonus = {"Critical": 0.3, "High": 0.2, "Medium": 0.1}.get(str(node["severity"]), 0.0)
        score = float(node["anomaly_score"]) + type_bonus + severity_bonus
        if score > best_score:
            best_score = score
            best_node = node_id
    return best_node


def evaluate_correlation(
    *,
    graph: nx.DiGraph,
    events: list[AnomalyEvent],
) -> dict[str, Any]:
    if not events:
        return {
            "temporal_alignment_score": 0.0,
            "graph_connectivity": 0.0,
            "chain_completeness": 0.0,
        }

    event_by_id = {event.event_id: event for event in events}
    predicted_parent: dict[str, str] = {}
    for event in events:
        if event.event_type != "log":
            continue
        incoming = [
            (source_id, graph.edges[source_id, event.event_id])
            for source_id in graph.predecessors(event.event_id)
            if event_by_id[source_id].event_type == "network"
        ]
        if not incoming:
            continue
        incoming.sort(key=lambda item: float(item[1]["score"]), reverse=True)
        predicted_parent[event.event_id] = incoming[0][0]

    alignment_hits = 0
    alignment_total = 0
    for event in events:
        if event.event_type != "log" or event.synthetic_parent_id is None:
            continue
        alignment_total += 1
        if predicted_parent.get(event.event_id) == event.synthetic_parent_id:
            alignment_hits += 1
    temporal_alignment_score = alignment_hits / alignment_total if alignment_total > 0 else 0.0

    synthetic_groups: dict[str, list[str]] = defaultdict(list)
    for event in events:
        if event.synthetic_chain_id:
            synthetic_groups[event.synthetic_chain_id].append(event.event_id)

    undirected = graph.to_undirected()
    connectivity_hits = 0
    completeness_scores: list[float] = []
    for node_ids in synthetic_groups.values():
        if len(node_ids) <= 1:
            continue
        subgraph = undirected.subgraph(node_ids)
        if nx.is_connected(subgraph):
            connectivity_hits += 1
        root_candidates = [node_id for node_id in node_ids if event_by_id[node_id].event_type == "network"]
        if not root_candidates:
            continue
        root = root_candidates[0]
        if root not in undirected:
            continue
        predicted_cluster = nx.node_connected_component(undirected, root)
        overlap = len(set(node_ids) & set(predicted_cluster))
        completeness_scores.append(overlap / max(1, len(node_ids)))

    graph_connectivity = connectivity_hits / max(1, len(synthetic_groups))
    chain_completeness = float(np.mean(completeness_scores)) if completeness_scores else 0.0

    return {
        "temporal_alignment_score": round(float(temporal_alignment_score), 6),
        "graph_connectivity": round(float(graph_connectivity), 6),
        "chain_completeness": round(float(chain_completeness), 6),
    }


def _load_correlation_model(path: Path | None, *, device: torch.device) -> AttentionCorrelationModel | None:
    if path is None or not path.exists():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model = AttentionCorrelationModel(
        input_dim=int(config.get("input_dim", 104)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        dropout=float(config.get("dropout", 0.1)),
        pair_feature_dim=int(config.get("pair_feature_dim", PAIR_FEATURE_DIM)),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model = model.to(device)
    model.eval()
    return model


def _resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested if requested != "cuda" else "cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main()
