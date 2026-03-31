"""
====================================
DataFlow-KG: KGGraphRAGSubgraphRetrieval
====================================

License:
    MIT License
"""

from collections import defaultdict, deque
from typing import List
import pandas as pd
from tqdm import tqdm

from dataflow import get_logger
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC


@OPERATOR_REGISTRY.register()
class KGGraphRAGSubgraphRetrieval(OperatorABC):
    """
    Graph-RAG subgraph retrieval operator (multi-question per row version).

    Input columns:
        - question: List[str]  # 单个行包含多个问题
        - entities: List[List[str]]  # 每个子列表对应一个问题的实体
        - relations: List[List[str]]
        - triplet: List[str]

    Output columns:
        - subgraph_prompt: List[str]  # 每个问题对应一个prompt
    """

    def __init__(self, hop: int = 1):
        self.hop = hop
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGGraphRAGSubgraphRetrieval 用于围绕问题实体检索子图并生成 GraphRAG 提示词。",
                "输入: question + entities + triplet; 输出: subgraph_prompt",
            )
        return (
            "KGGraphRAGSubgraphRetrieval is used to retrieve subgraphs around question entities and build GraphRAG prompts.",
            "Input: question + entities + triplet; Output: subgraph_prompt",
        )

    # --------------------------------------------------
    # Triplet parsing
    # --------------------------------------------------
    @staticmethod
    def _parse_triplet(triplet: str):
        """
        Parse:
        "<subj> Henry <obj> Maria Rodriguez <rel> is_trained_by"
        -> (Henry, is_trained_by, Maria Rodriguez)
        """
        try:
            subj = triplet.split("<subj>")[1].split("<obj>")[0].strip()
            obj = triplet.split("<obj>")[1].split("<rel>")[0].strip()
            rel = triplet.split("<rel>")[1].strip()
            return subj, rel, obj
        except Exception:
            return None, None, None

    # --------------------------------------------------
    # Entity catalog induction
    # --------------------------------------------------
    @classmethod
    def _build_entity_catalog(cls, triplets: List[str]):
        entities = set()
        for t in triplets:
            h, _, o = cls._parse_triplet(t)
            if h:
                entities.add(h)
            if o:
                entities.add(o)
        return entities

    # --------------------------------------------------
    # k-hop BFS subgraph sampling
    # --------------------------------------------------
    @classmethod
    def _k_hop_subgraph(
        cls,
        triplets: List[str],
        start_entity: str,
        hop: int,
    ):
        adj = defaultdict(list)

        for t in triplets:
            h, r, o = cls._parse_triplet(t)
            if h is None:
                continue
            adj[h].append((o, t))
            adj[o].append((h, t))  # treat as undirected

        visited_entities = {start_entity}
        visited_triplets = set()
        queue = deque([(start_entity, 0)])

        while queue:
            cur, depth = queue.popleft()
            if depth == hop:
                continue

            for nxt, raw_t in adj.get(cur, []):
                visited_triplets.add(raw_t)
                if nxt not in visited_entities:
                    visited_entities.add(nxt)
                    queue.append((nxt, depth + 1))

        return list(visited_triplets)

    # --------------------------------------------------
    # Prompt construction (单个问题)
    # --------------------------------------------------
    def _build_single_prompt(
        self,
        question: str,
        entities: List[str],
        triplets: List[str],
    ) -> str:
        # 标准化实体列表（处理嵌套列表情况）
        normalized_entities = []
        for e in entities:
            if isinstance(e, list):
                normalized_entities.extend(e)
            else:
                normalized_entities.append(e)
        
        # 去重，避免重复处理相同实体
        normalized_entities = list(set(normalized_entities))

        # 1. 构建KG中的实体目录
        entity_catalog = self._build_entity_catalog(triplets)

        # 2. 种子实体 = 提取的实体 ∩ KG中的实体
        seed_entities = [
            e for e in normalized_entities if e in entity_catalog
        ]

        # 降级策略：无匹配实体时取KG中第一个实体
        if not seed_entities and entity_catalog:
            seed_entities = [next(iter(entity_catalog))]

        # 3. 采样子图
        subgraphs = {}
        for ent in seed_entities:
            subgraphs[ent] = self._k_hop_subgraph(
                triplets,
                start_entity=ent,
                hop=self.hop,
            )

        # 4. 组装prompt
        lines = []
        lines.append("You are given a question and relevant knowledge graph facts.")
        lines.append("Use ONLY the provided facts to answer the question.\n")

        lines.append("Question:")
        lines.append(question.strip() + "\n")

        for ent, sg in subgraphs.items():
            lines.append(f"Subgraph centered at [{ent}]:")
            if sg:  # 避免空行
                for t in sg:
                    lines.append(f"- {t}")
            else:
                lines.append("- No relevant triplets found")
            lines.append("")

        lines.append("Answer the question based on the above knowledge graph subgraphs.")

        return "\n".join(lines)

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------
    def _validate_dataframe(self, df: pd.DataFrame):
        required = ["question", "entities", "triplet"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        if "subgraph_prompt" in df.columns:
            raise ValueError("Output column already exists: subgraph_prompt")

    # --------------------------------------------------
    # Run (适配多问题每行的格式) - 修正版
    # --------------------------------------------------
    def run(
        self,
        storage: DataFlowStorage = None,
        output_key: str = "subgraph_prompt",
    ) -> List[List[str]]:
        # 检查 storage
        if storage is None:
            raise ValueError("storage parameter cannot be None")
        
        df = storage.read("dataframe")
        self._validate_dataframe(df)

        all_prompts = []

        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc="Graph-RAG subgraph retrieval",
        ):
            # ========== 适配你的输入格式 ==========
            # 1. 提取问题
            questions = row.get("question", [])
            if questions is None or not isinstance(questions, list):
                questions = []

            # 2. 提取实体列表
            entities_list = row.get("entities", [])
            if entities_list is None or not isinstance(entities_list, list):
                entities_list = [[] for _ in questions]

            # 3. 提取三元组
            triplets = row.get("triplet", [])
            if triplets is None or not isinstance(triplets, list):
                triplets = []

            # 4. 确保问题数量和实体列表数量匹配
            max_len = max(len(questions), len(entities_list))
            if len(questions) < max_len:
                questions += [""] * (max_len - len(questions))
            if len(entities_list) < max_len:
                entities_list += [[] for _ in range(max_len - len(entities_list))]

            # 5. 为每个问题生成 prompt
            row_prompts = []
            for q, ents in zip(questions, entities_list):
                if not isinstance(q, str) or not q.strip():  # 跳过空问题
                    row_prompts.append("")
                    continue
                if not isinstance(ents, list):
                    ents = [str(ents)]  # 避免非列表情况
                prompt = self._build_single_prompt(
                    question=q,
                    entities=ents,
                    triplets=triplets,
                )
                row_prompts.append(prompt)

            all_prompts.append(row_prompts)
            # ========== 适配结束 ==========

        # 写入 DataFrame
        df[output_key] = all_prompts

        # 保存结果
        output_file = storage.write(df)
        self.logger.info(f"Graph-RAG prompts saved to {output_file}")

        return [output_key]
