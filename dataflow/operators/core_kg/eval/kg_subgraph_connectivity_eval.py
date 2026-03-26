import pandas as pd
import networkx as nx
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from typing import List
import re


@OPERATOR_REGISTRY.register()
class KGSubgraphConnectivityEvaluator(OperatorABC):
    """
    Compute connectivity metrics for KG subgraphs.

    Input column:
        subgraph : List[str]

    Triple format:
        "<subj> Henry <obj> Lucy <rel> is_inspired_by"

    Output columns:
        vertex_connectivity
        edge_connectivity
        global_efficiency
    """

    def __init__(self, merge_to_input: bool = False):
        self.logger = get_logger()
        self.merge_to_input = merge_to_input

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGSubgraphConnectivityEvaluator 用于评估知识图谱子图的连通性指标。",
                "计算每个子图的点连通度、边连通度以及全局效率。",
                "输入列 subgraph 为子图三元组字符串列表，输出列 vertex_connectivity（点连通度）、edge_connectivity（边连通度）和 global_efficiency（全局效率）分别写入 DataFrame。",
            )
        return (
            "KGSubgraphConnectivityEvaluator computes connectivity metrics for KG subgraphs.",
            "Calculates vertex connectivity, edge connectivity, and global efficiency for each subgraph.",
            "Takes subgraph (List[str] of relation triples) as input and outputs vertex_connectivity, edge_connectivity, and global_efficiency columns.",
        )

    def _validate_dataframe(self, df: pd.DataFrame, input_key: str):
        if input_key not in df.columns:
            raise ValueError(f"Missing required column: {input_key}")

    def _parse_triple(self, triple: str):
        """
        Parse triple string.
        """
        pattern = r"<subj>\s*(.*?)\s*<obj>\s*(.*?)\s*<rel>\s*(.*)"
        match = re.match(pattern, triple)

        if match:
            subj = match.group(1).strip()
            obj = match.group(2).strip()
            rel = match.group(3).strip()
            return subj, obj, rel

        return None, None, None

    def _build_graph(self, subgraph: List[str]):
        """
        Build graph from triple list.
        """
        G = nx.Graph()

        for triple in subgraph:
            subj, obj, rel = self._parse_triple(triple)

            if subj is None:
                continue

            G.add_node(subj)
            G.add_node(obj)

            # ignore relation type for connectivity
            G.add_edge(subj, obj)

        return G

    def _compute_connectivity_metrics(self, subgraph: List[str]):
        """
        Compute connectivity metrics.
        """

        G = self._build_graph(subgraph)

        if G.number_of_nodes() <= 1:
            return 0, 0, 0.0

        try:
            vertex_conn = nx.node_connectivity(G)
        except:
            vertex_conn = 0

        try:
            edge_conn = nx.edge_connectivity(G)
        except:
            edge_conn = 0

        try:
            efficiency = nx.global_efficiency(G)
        except:
            efficiency = 0.0

        return vertex_conn, edge_conn, efficiency

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "subgraph",
        output_key1: str = "vertex_connectivity",
        output_key2: str = "edge_connectivity",
        output_key3: str = "global_efficiency",
    ):
        """
        Run connectivity evaluation.
        """

        df = storage.read("dataframe")
        self._validate_dataframe(df, input_key)

        self.logger.info("Computing connectivity metrics...")

        vertex_list = []
        edge_list = []
        efficiency_list = []

        for subgraph in df[input_key]:

            if not isinstance(subgraph, list):
                vertex_list.append(0)
                edge_list.append(0)
                efficiency_list.append(0.0)
                continue

            v, e, eff = self._compute_connectivity_metrics(subgraph)

            vertex_list.append(v)
            edge_list.append(e)
            efficiency_list.append(eff)

        df[output_key1] = vertex_list
        df[output_key2] = edge_list
        df[output_key3] = efficiency_list

        output_file = storage.write(df)

        self.logger.info(f"Connectivity metrics saved to {output_file}")

        return [output_key1, output_key2, output_key3]