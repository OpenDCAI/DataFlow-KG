"""
====================================
DataFlow-KG:
====================================

Author: Zhengpin Li
Affiliation: Peking University
Email: zpli@pku.edu.cn
Created: 2026-01-27

License:
    MIT License
"""

import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Union
import json
from tqdm import tqdm
import re
import networkx as nx
from pyvis.network import Network
from collections import Counter

from dataflow.core.prompt import prompt_restrict, DIYPromptABC

import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
import random
from typing import List
import re
import networkx as nx
from pyvis.network import Network
from collections import Counter


@OPERATOR_REGISTRY.register()
class KGRelationTripleVisualization(OperatorABC):
    """
    KGTripleVisualization visualizes knowledge graph triples as an interactive graph.

    It converts entity–relation–entity triples into a directed graph and renders
    the graph as an HTML file using PyVis for interactive inspection.
    """

    def __init__(
        self,
        llm_serving: LLMServingABC = None,
        seed: int = 0,
        lang: str = "en",
    ):
        self.rng = random.Random(seed)
        self.lang = lang
        self.logger = get_logger()

        # Pattern for parsing entity–relation–entity triples
        self.triplet_pattern = re.compile(
            r"<subj>\s*(.+?)\s*<obj>\s*(.+?)\s*<rel>\s*(.+?)$"
        )

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGTripleVisualization 用于将知识图谱三元组可视化为交互式图结构。",
                "该算子基于实体关系三元组构建有向图，并输出 HTML 可视化结果。",
                "适用于知识图谱分析、调试与结构检查。",
            )
        else:
            return (
                "KGTripleVisualization visualizes knowledge graph triples as an interactive graph.",
                "It builds a directed graph from entity–relation–entity triples.",
                "The output is an HTML file for inspecting graph structure and connectivity.",
            )

    def _visualize_kg_with_pyvis(
        self,
        triple_lists: List[List[str]],
        output_html: str = "/data/zhengpinli/DataFlow/dataflow/operators/knowledge_graph/eval/kg_visualization.html",
        notebook: bool = False,
    ):
        """
        Render a list of triple lists as an interactive knowledge graph.
        """
        # Flatten nested triple lists
        triples = [t for sublist in triple_lists for t in sublist]
        if not triples:
            self.logger.warning("Empty graph: no triples to visualize.")
            return None

        edges = []
        entity_counter = Counter()

        # Parse triples and collect edges
        for t in triples:
            match = self.triplet_pattern.match(t.strip())
            if not match:
                self.logger.warning(f"Failed to parse triple: {t}")
                continue

            relation, subj, obj = match.groups()
            edges.append((subj, obj, relation))
            entity_counter[subj] += 1
            entity_counter[obj] += 1

        if not edges:
            self.logger.warning("No valid triples after parsing.")
            return None

        # Build directed graph
        G = nx.DiGraph()
        for s, o, r in edges:
            G.add_edge(s, o, label=r)

        # Initialize PyVis network
        net_vis = Network(
            height="750px",
            width="100%",
            directed=True,
            notebook=notebook,
        )
        net_vis.barnes_hut()

        # Add nodes with size scaled by frequency
        for node in G.nodes():
            net_vis.add_node(
                node,
                label=node,
                size=10 + entity_counter[node] * 5,
                title=f"Entity: {node}<br>Frequency: {entity_counter[node]}",
                font={"size": 48, "face": "arial"},
            )

        # Add directed edges with relation labels
        for s, o, data in G.edges(data=True):
            net_vis.add_edge(
                s,
                o,
                label=data["label"],
                title=data["label"],
                arrows="to",
                font={"size": 48, "face": "arial"},
            )

        # Write visualization to HTML
        net_vis.write_html(output_html, open_browser=False, notebook=False)
        self.logger.info(f"Knowledge graph visualization saved to: {output_html}")

        return net_vis

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        if self.input_key not in dataframe.columns:
            raise ValueError(f"Missing required column: {self.input_key}")
        if self.output_key in dataframe.columns:
            raise ValueError(
                f"Column '{self.output_key}' already exists and would be overwritten"
            )

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "triple",
        output_key: str = "kg_visualization",
    ):
        """
        Execute knowledge graph visualization.
        """
        self.input_key = input_key
        self.output_key = output_key

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        triple_lists = dataframe[self.input_key].tolist()
        self._visualize_kg_with_pyvis(triple_lists)