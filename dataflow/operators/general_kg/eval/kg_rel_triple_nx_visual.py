import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
import os

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
                "KGRelationTripleVisualization 用于将知识图谱三元组可视化为交互式图结构。",
                "该算子基于实体关系三元组构建有向图，并使用 PyVis 渲染为 HTML 文件。",
                "输入列 triple 为关系三元组字符串列表，输出列 kg_visualization 为 HTML 可视化文件路径。",
            )
        else:
            return (
                "KGRelationTripleVisualization visualizes knowledge graph triples as an interactive graph.",
                "Builds a directed graph from entity–relation–entity triples and renders it as an HTML file using PyVis.",
                "Takes triple (List[str] of relation triples) as input column and outputs kg_visualization (path to the generated HTML file).",
            )

    def _visualize_kg_with_pyvis(
        self,
        triple_lists: List[List[str]],
        output_html: str = "",
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

        if not output_html:
            output_html = os.path.abspath("kg_visualization.html")
        output_dir = os.path.dirname(output_html)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        edges = []
        entity_counter = Counter()

        # Parse triples and collect edges
        for t in triples:
            match = self.triplet_pattern.match(t.strip())
            if not match:
                self.logger.warning(f"Failed to parse triple: {t}")
                continue

            subj, obj, relation = match.groups()
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
        output_html: str = "",
    ):
        """
        Execute knowledge graph visualization.
        """
        self.input_key = input_key
        self.output_key = output_key

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        triple_lists = dataframe[self.input_key].tolist()
        self._visualize_kg_with_pyvis(triple_lists, output_html=output_html)
        dataframe[self.output_key] = output_html if output_html else os.path.abspath("kg_visualization.html")
        storage.write(dataframe)
        return [output_key]
