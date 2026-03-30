import json
from typing import List, Dict, Any
from tqdm import tqdm

from dataflow import get_logger
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.prompts.diverse_kg.cskg import CSKGTripleRationalePrompt


@OPERATOR_REGISTRY.register()
class CSKGTripleRationaleEvaluator(OperatorABC):

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "CSKGTripleRationaleEvaluator 用于评估常识知识图谱（CSKG）三元组的合理性得分（rationale scores）。",
                "输入为三元组列表（默认字段 triple），输出为大模型评估的合理性得分列表（默认字段 rationale_scores）。"
            )
        else:
            return (
                "CSKGTripleRationaleEvaluator evaluates the rationale scores of commonsense knowledge graph (CSKG) triples.",
                "Input: lists of triples. Output: corresponding rationale scores evaluated by LLM."
            )

    def __init__(
        self,
        llm_serving: LLMServingABC,
        lang: str = "en"
    ):
        super().__init__()

        self.logger = get_logger()

        if not isinstance(llm_serving, LLMServingABC):
            raise TypeError("llm_serving must be LLMServingABC")

        self.llm_serving = llm_serving
        self.prompt_manager = CSKGTripleRationalePrompt(lang)

    def _safe_parse_json(self, response: str) -> Dict[str, Any]:

        clean = response.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(clean)
        except:
            # hy修复：更正兜底的返回字段
            return {"rationale_scores": []}


    def process_batch(self, records: List[Dict[str, Any]]):

        results = []

        # hy修复：更正进度条文案
        for row in tqdm(records, desc="Triple Rationale Eval"):

            # hy修复：将错写的 qa_pairs 统一更正为 triples
            triples = row.get("triple", [])

            if isinstance(triples, str):
                try:
                    triples = json.loads(triples)
                except:
                    triples = []

            if not triples:
                results.append({
                    "rationale_scores": []
                })
                continue

            try:

                system_prompt = self.prompt_manager.build_system_prompt()
                user_prompt = self.prompt_manager.build_prompt(triples)

                response = self.llm_serving.generate_from_input(
                    user_inputs=[user_prompt],
                    system_prompt=system_prompt
                )[0]

                data = self._safe_parse_json(response)

                scores = data.get("rationale_scores", [])

                results.append({
                    "rationale_scores": scores
                })

            except Exception as e:

                self.logger.error(f"LLM Error: {e}")

                results.append({
                    "rationale_scores": []
                })

        return results


    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "triple",
        output_key: str = "rationale_scores"
    ):

        if storage is None:
            raise ValueError("Storage required.")

        df = storage.read("dataframe")

        records = []

        for _, r in df.iterrows():

            records.append({
                "triple": r.get(input_key, [])
            })

        outputs = self.process_batch(records)

        df[output_key] = [
            o.get(output_key, [])
            for o in outputs
        ]

        out_file = storage.write(df)

        # hy修复：更正日志输出文案
        self.logger.info(
            f"Saved triple rationale scores to {out_file}"
        )

        return [output_key]