import json
import random
from typing import List, Dict, Any
from tqdm import tqdm

from dataflow import get_logger
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.prompts.core_kg.rel_triple_eval import KGRelationConsistencyEvaluationPrompt


class KGRelationTripleConsistencyEvaluator(OperatorABC):

    def __init__(
        self,
        llm_serving: LLMServingABC,
        sample_rate: float = 1,
        max_samples: int = 10,
        lang: str = "en"
    ):
        super().__init__()
        self.logger = get_logger()

        if not isinstance(llm_serving, LLMServingABC):
            raise TypeError("llm_serving must be an instance of LLMServingABC")

        self.llm_serving = llm_serving
        self.sample_rate = sample_rate
        self.max_samples = max_samples
        self.prompt_manager = KGRelationConsistencyEvaluationPrompt(lang)

    # =========================
    # Description
    # =========================
    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGRelationTripleConsistencyEvaluator 用于评估知识图谱关系三元组的逻辑一致性。",
                "通过采样三元组并利用 LLM 进行上下文一致性判断。",
                "输入列 triple 为关系三元组字符串列表，输出列 logical_consistency_score 为一致性得分（0~1 浮点数），evaluated_sample_indices 为采样三元组的索引列表。"
            )
        else:
            return (
                "KGRelationTripleConsistencyEvaluator evaluates logical consistency of KG relation triples.",
                "Samples triples and uses an LLM to judge contextual consistency.",
                "Takes triple (List[str] of relation triples) as input and outputs logical_consistency_score (float in [0,1]) and evaluated_sample_indices (List of sampled indices)."
            )

    # ============================================================
    # Parsing
    # ============================================================

    def _parse_relation_triple(self, triple_str: str) -> List[str]:
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
            self.logger.warning(f"Failed to parse triple: {triple_str}, error: {e}")
            return []

    # ============================================================
    # Graph
    # ============================================================

    def _build_graph(self, triples: List[str]):
        G = SimpleGraph()

        for triple_str in triples:
            parsed = self._parse_relation_triple(triple_str)
            if len(parsed) >= 3:
                s, p, o = parsed
                if s and o:
                    G.add_edge(s, o, relation=p, subject=s, object=o)

        return G

    def _get_local_context(self, G, u, v, limit=5):

        context_facts = []

        # subject neighbors
        for n in G.neighbors(u):
            if n == v:
                continue
            if G.has_edge(u, n):
                data = G.get_edge_data(u, n)
                rel = data.get("relation", "related_to")
                context_facts.append(f"- {u} {rel} {n}")
                if len(context_facts) >= limit:
                    break

        # object neighbors
        for n in G.neighbors(v):
            if n == u:
                continue
            if G.has_edge(v, n):
                data = G.get_edge_data(v, n)
                rel = data.get("relation", "related_to")
                context_facts.append(f"- {v} {rel} {n}")
                if len(context_facts) >= limit * 2:
                    break

        if not context_facts:
            return "No specific context available (Isolated Edge)."

        return "\n".join(context_facts)

    # ============================================================
    # LLM
    # ============================================================

    def _safe_parse_json(self, response: str) -> Dict[str, Any]:
        clean_text = response.strip()

        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(clean_text)
        except:
            return {}

    def _check_consistency(self, context_desc, subject, object_entity, relation):

        if context_desc == "No specific context available (Isolated Edge).":
            return 0

        system_prompt = self.prompt_manager.build_system_prompt()
        user_prompt = self.prompt_manager.build_prompt(
            context_desc=context_desc,
            subj=subject,
            obj=object_entity,
            relation=relation
        )

        try:
            responses = self.llm_serving.generate_from_input(
                user_inputs=[user_prompt],
                system_prompt=system_prompt
            )

            if not responses:
                return 0

            response = responses[0]
            res_json = self._safe_parse_json(response)
            judgment = res_json.get("judgment", "").upper()

            if judgment == "CONSISTENT":
                return 1
            elif judgment == "INCONSISTENT":
                return 0

            response_upper = response.upper()
            if "CONSISTENT" in response_upper and "INCONSISTENT" not in response_upper:
                return 1

            return 0

        except Exception as e:
            self.logger.error(f"LLM failed: {e}")
            return 0

    # ============================================================
    # Core Logic
    # ============================================================

    def process_batch(self, batch_data: List[Dict[str, Any]]):

        results = []

        for row_idx, row in enumerate(tqdm(batch_data)):

            test_triples = row.get("test_triple", None)
            triples = row.get("triple", [])

            # ===============================
            # 情况1：存在 test_triple
            # ===============================
            if test_triples is not None:

                if isinstance(test_triples, str):
                    try:
                        test_triples = json.loads(test_triples)
                    except:
                        test_triples = []

                if not isinstance(test_triples, list) or not test_triples:
                    results.append({"logical_consistency_score": 0.0})
                    continue

                G = self._build_graph(test_triples)
                edges = G.edges(data=True)

                if not edges:
                    results.append({"logical_consistency_score": 0.0})
                    continue

                hits = 0
                total = 0

                for (u, v, data) in edges:
                    rel = data.get("relation")
                    if not rel:
                        continue

                    context = self._get_local_context(G, u, v)
                    hits += self._check_consistency(context, u, v, rel)
                    total += 1

                score = hits / total if total > 0 else 0.0
                results.append({"logical_consistency_score": score})

            # ===============================
            # 情况2：采样 triple
            # ===============================
            else:

                if isinstance(triples, str):
                    try:
                        triples = json.loads(triples)
                    except:
                        triples = []

                if not isinstance(triples, list) or not triples:
                    results.append({
                        "logical_consistency_score": 0.0,
                        "evaluated_sample_indices": []
                    })
                    continue

                G = self._build_graph(triples)
                edges = G.edges(data=True)

                if not edges:
                    results.append({
                        "logical_consistency_score": 0.0,
                        "evaluated_sample_indices": []
                    })
                    continue

                n_sample = max(
                    1,
                    min(int(len(edges) * self.sample_rate), self.max_samples)
                )

                if len(edges) > n_sample:
                    sampled_indices = random.sample(range(len(edges)), n_sample)
                else:
                    sampled_indices = list(range(len(edges)))

                hits = 0
                total = 0

                for idx in sampled_indices:
                    u, v, data = edges[idx]
                    rel = data.get("relation")
                    if not rel:
                        continue

                    context = self._get_local_context(G, u, v)
                    hits += self._check_consistency(context, u, v, rel)
                    total += 1

                score = hits / total if total > 0 else 0.0

                results.append({
                    "logical_consistency_score": score,
                    "evaluated_sample_indices": sampled_indices
                })

        return results

    # ============================================================
    # Run
    # ============================================================

    def run(self, storage: DataFlowStorage = None, input_key: str = "triple"):

        if storage is None:
            raise ValueError("Storage is required.")

        df = storage.read("dataframe")

        records = []
        for _, r in df.iterrows():
            records.append({
                "triple": r.get(input_key, []),
                "test_triple": r.get("test_triple", None)
            })

        outputs = self.process_batch(records)

        df["logical_consistency_score"] = [
            o.get("logical_consistency_score", 0.0) for o in outputs
        ]

        df["evaluated_sample_indices"] = [
            o.get("evaluated_sample_indices", None) for o in outputs
        ]

        out_file = storage.write(df)

        self.logger.info(
            f"Consistency Eval Done. "
            f"Avg Score: {df['logical_consistency_score'].mean():.4f}. "
            f"Saved to {out_file}"
        )

        return ["logical_consistency_score", "evaluated_sample_indices"]


# ============================================================
# SimpleGraph
# ============================================================

class SimpleGraph:

    def __init__(self):
        self.adj: Dict[Any, Dict[Any, Dict[str, Any]]] = {}

    def add_edge(self, u, v, **attrs):
        if u not in self.adj:
            self.adj[u] = {}
        if v not in self.adj:
            self.adj[v] = {}
        self.adj[u][v] = attrs
        self.adj[v][u] = attrs

    def neighbors(self, node):
        return list(self.adj.get(node, {}).keys())

    def has_edge(self, u, v):
        return u in self.adj and v in self.adj[u]

    def get_edge_data(self, u, v):
        if self.has_edge(u, v):
            return self.adj[u][v]
        return {}

    def edges(self, data=False):
        seen = set()
        result = []

        for u in self.adj:
            for v, attrs in self.adj[u].items():
                key = tuple(sorted([str(u), str(v)]))
                if key not in seen:
                    seen.add(key)
                    if data:
                        result.append((u, v, attrs))
                    else:
                        result.append((u, v))

        return result