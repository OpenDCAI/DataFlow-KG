import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from typing import List
import re


@OPERATOR_REGISTRY.register()
class KGSubgraphScaleEvaluator(OperatorABC):
    """
    Compute structural statistics for KG subgraphs.

    Input column:
        subgraph : List[str]

    Example triple format:
        "<subj> Henry <obj> Lucy <rel> is_inspired_by"

    Output columns:
        num_nodes
        num_edges
        density
    """

    def __init__(self, merge_to_input: bool = False):
        self.logger = get_logger()
        self.merge_to_input = merge_to_input

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGSubgraphScaleEvaluator 用于统计知识图谱子图的结构规模指标。",
                "计算每个子图的节点数、边数及密度。",
                "输入列 subgraph 为子图三元组字符串列表，输出列 num_nodes（节点数）、num_edges（边数）和 density（密度）分别写入 DataFrame。",
            )
        return (
            "KGSubgraphScaleEvaluator computes structural scale metrics for KG subgraphs.",
            "Calculates the number of nodes, number of edges, and density for each subgraph.",
            "Takes subgraph (List[str] of relation triples) as input and outputs num_nodes, num_edges, and density columns.",
        )

    def _validate_dataframe(self, df: pd.DataFrame, input_key: str):
        if input_key not in df.columns:
            raise ValueError(f"Missing required column: {input_key}")

    def _parse_triple(self, triple: str):
        """
        Parse triple string into (subject, object, relation)
        """
        pattern = r"<subj>\s*(.*?)\s*<obj>\s*(.*?)\s*<rel>\s*(.*)"
        match = re.match(pattern, triple)
        if match:
            subj = match.group(1).strip()
            obj = match.group(2).strip()
            rel = match.group(3).strip()
            return subj, obj, rel
        return None, None, None

    def _compute_statistics(self, subgraph: List[str]):
        """
        Compute node count, edge count, density.
        """
        nodes = set()
        edges = []

        for triple in subgraph:
            subj, obj, rel = self._parse_triple(triple)

            if subj is None:
                continue

            nodes.add(subj)
            nodes.add(obj)

            edges.append((subj, rel, obj))

        num_nodes = len(nodes)
        num_edges = len(edges)

        if num_nodes <= 1:
            density = 0.0
        else:
            density = num_edges / (num_nodes * (num_nodes - 1))

        return num_nodes, num_edges, density

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "subgraph",
        output_key1: str = "num_nodes",
        output_key2: str = "num_edges",
        output_key3: str = "density",
    ):
        """
        Run subgraph statistics computation.
        """

        df = storage.read("dataframe")
        self._validate_dataframe(df, input_key)

        self.logger.info("Computing subgraph statistics...")

        num_nodes_list = []
        num_edges_list = []
        density_list = []

        for subgraph in df[input_key]:

            if not isinstance(subgraph, list):
                num_nodes_list.append(0)
                num_edges_list.append(0)
                density_list.append(0.0)
                continue

            n, e, d = self._compute_statistics(subgraph)

            num_nodes_list.append(n)
            num_edges_list.append(e)
            density_list.append(d)

        df[output_key1] = num_nodes_list
        df[output_key2] = num_edges_list
        df[output_key3] = density_list

        output_file = storage.write(df)

        self.logger.info(f"Statistics saved to {output_file}")

        return [output_key1, output_key2, output_key3]