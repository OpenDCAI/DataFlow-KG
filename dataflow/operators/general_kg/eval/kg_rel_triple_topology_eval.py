import json
import networkx as nx  # pyright: ignore[reportMissingModuleSource]
from typing import List, Dict, Any
from tqdm import tqdm

from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage


class KGRelationTripleTopologyEvaluator(OperatorABC):
    """
    Evaluates the Topological Structure of a Knowledge Graph
    constructed from entity-relation triples only.

    Input:
        - triple: List[str] in format:
          "<subj> X <obj> Y <rel> Z"

    Metrics:
        1. lcc_ratio
        2. structure_avg_degree
        3. fragmentation_score
        4. num_components
        5. node_count
        6. edge_count
    """

    def __init__(self):
        super().__init__()
        self.logger = get_logger()

    # =========================
    # Description
    # =========================
    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGRelationTripleTopologyEvaluator 用于评估知识图谱的拓扑结构特征。",
                "计算最大连通分量比例、平均度数、碎片化程度等图拓扑指标。",
                "输入列 triple 为关系三元组字符串列表，输出列包括 lcc_ratio（最大连通分量比例）、structure_avg_degree（平均度数）、fragmentation_score（碎片化分数）、num_components（连通分量数）、node_count（节点数）、edge_count（边数）。"
            )
        else:
            return (
                "KGRelationTripleTopologyEvaluator evaluates the topological structure of a KG.",
                "Computes LCC ratio, average degree, fragmentation score, and component metrics.",
                "Takes triple (List[str] of relation triples) as input and outputs lcc_ratio, structure_avg_degree, fragmentation_score, num_components, node_count, and edge_count."
            )

    # ============================================================
    # Parsing Relation Triple Only
    # ============================================================

    def _parse_relation_triple(self, triple_str: str) -> List[str]:
        """
        Parse relation triple:
        "<subj> X <obj> Y <rel> Z"
        """
        try:
            if not isinstance(triple_str, str):
                return []

            if "<subj>" not in triple_str or "<obj>" not in triple_str or "<rel>" not in triple_str:
                return []

            subj_parts = triple_str.split("<subj>")
            if len(subj_parts) != 2:
                return []

            obj_parts = subj_parts[1].split("<obj>")
            if len(obj_parts) != 2:
                return []

            subject = obj_parts[0].strip()

            rel_parts = obj_parts[1].split("<rel>")
            if len(rel_parts) != 2:
                return []

            obj = rel_parts[0].strip()
            predicate = rel_parts[1].strip()

            return [subject, predicate, obj]

        except Exception as e:
            self.logger.warning(f"Failed to parse relation triple: {triple_str}, error: {e}")
            return []

    # ============================================================
    # Graph Construction
    # ============================================================

    def _build_graph(self, triples: List[str]) -> nx.Graph:
        """
        Build undirected graph from relation triples.
        """
        G = nx.Graph()

        for triple_str in triples:
            parsed = self._parse_relation_triple(triple_str)
            if len(parsed) >= 3:
                s, p, o = parsed
                if s and o:
                    G.add_edge(s, o, relation=p)

        return G

    # ============================================================
    # Topology Metrics
    # ============================================================

    def _evaluate_graph(self, G: nx.Graph) -> Dict[str, Any]:

        num_nodes = len(G.nodes)
        num_edges = len(G.edges)

        if num_nodes == 0:
            return {
                "lcc_ratio": 0.0,
                "structure_avg_degree": 0.0,
                "fragmentation_score": 1.0,
                "num_components": 0,
                "node_count": 0,
                "edge_count": 0
            }

        components = list(nx.connected_components(G))
        num_components = len(components)

        largest_cc = max(components, key=len) if components else set()
        lcc_ratio = len(largest_cc) / num_nodes

        avg_degree = (2 * num_edges) / num_nodes

        if num_nodes > 1:
            fragmentation_score = (num_components - 1) / (num_nodes - 1)
        else:
            fragmentation_score = 0.0

        return {
            "lcc_ratio": round(lcc_ratio, 4),
            "structure_avg_degree": round(avg_degree, 4),
            "fragmentation_score": round(fragmentation_score, 4),
            "num_components": num_components,
            "node_count": num_nodes,
            "edge_count": num_edges
        }

    # ============================================================
    # Batch Processing
    # ============================================================

    def process_batch(self, dataframe_subset: List[Dict[str, Any]]):

        results = []

        for row in tqdm(dataframe_subset, desc="Evaluating Graph Structure"):

            triples = row.get("triple", [])

            if isinstance(triples, str):
                try:
                    triples = json.loads(triples)
                except:
                    triples = []

            if not isinstance(triples, list):
                triples = []

            if not triples:
                results.append({
                    "lcc_ratio": 0.0,
                    "structure_avg_degree": 0.0,
                    "fragmentation_score": 1.0,
                    "num_components": 0,
                    "node_count": 0,
                    "edge_count": 0
                })
                continue

            G = self._build_graph(triples)
            metrics = self._evaluate_graph(G)

            results.append(metrics)

        return results

    # ============================================================
    # Run
    # ============================================================

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "triple"
    ):

        if storage is None:
            raise ValueError("Storage is required.")

        df = storage.read("dataframe")
        self.logger.info(f"Starting Graph Structure Eval on {len(df)} records.")

        records = []
        for _, r in df.iterrows():
            records.append({
                "triple": r.get(input_key, [])
            })

        outputs = self.process_batch(records)

        metric_keys = [
            "lcc_ratio",
            "structure_avg_degree",
            "fragmentation_score",
            "num_components",
            "node_count",
            "edge_count"
        ]

        for key in metric_keys:
            df[key] = [o.get(key, 0.0) for o in outputs]

        out_file = storage.write(df)

        self.logger.info(
            f"Structure Evaluation complete. "
            f"Avg LCC Ratio: {df['lcc_ratio'].mean():.4f}. "
            f"Saved to {out_file}"
        )

        return metric_keys