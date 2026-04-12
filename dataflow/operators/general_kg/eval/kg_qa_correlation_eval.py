import json
from typing import List, Dict, Any
from tqdm import tqdm

from dataflow import get_logger
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.prompts.core_kg.rel_triple_eval import KGQACorrelationPrompt


class KGQACorrelationEvaluator(OperatorABC):

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
        self.prompt_manager = KGQACorrelationPrompt(lang)

    # =========================
    # Description
    # =========================
    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGQACorrelationEvaluator 用于评估知识图谱 QA 对的相关性。",
                "使用 LLM 对 QA 对进行相关性打分。",
                "输入列 QA_pairs 为 QA 问答对列表，输出列 correlation_scores 为每对 QA 的相关性评分列表。"
            )
        else:
            return (
                "KGQACorrelationEvaluator evaluates the correlation of KG-derived QA pairs.",
                "Uses an LLM to score each QA pair on correlation.",
                "Takes QA_pairs (List of QA pairs) as input and outputs correlation_scores (List of scores per QA pair)."
            )

    # ============================================================
    # JSON Parse
    # ============================================================

    def _safe_parse_json(self, response: str) -> Dict[str, Any]:

        clean = response.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(clean)
        except:
            return {"correlation_scores": []}

    # ============================================================
    # Core Evaluation
    # ============================================================

    def process_batch(self, records: List[Dict[str, Any]]):

        results = []

        for row in tqdm(records, desc="QA Correlation Eval"):

            qa_pairs = row.get("QA_pairs", [])

            if isinstance(qa_pairs, str):
                try:
                    qa_pairs = json.loads(qa_pairs)
                except:
                    qa_pairs = []

            if not qa_pairs:
                results.append({
                    "correlation_scores": []
                })
                continue

            try:

                system_prompt = self.prompt_manager.build_system_prompt()
                user_prompt = self.prompt_manager.build_prompt(qa_pairs)

                response = self.llm_serving.generate_from_input(
                    user_inputs=[user_prompt],
                    system_prompt=system_prompt
                )[0]

                data = self._safe_parse_json(response)

                scores = data.get("correlation_scores", [])

                results.append({
                    "correlation_scores": scores
                })

            except Exception as e:

                self.logger.error(f"LLM Error: {e}")

                results.append({
                    "correlation_scores": []
                })

        return results

    # ============================================================
    # Run
    # ============================================================

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "QA_pairs",
        output_key: str = "correlation_scores"
    ):

        if storage is None:
            raise ValueError("Storage required.")

        df = storage.read("dataframe")

        records = []

        for _, r in df.iterrows():

            records.append({
                "QA_pairs": r.get(input_key, [])
            })

        outputs = self.process_batch(records)

        df[output_key] = [
            o.get(output_key, [])
            for o in outputs
        ]

        out_file = storage.write(df)

        self.logger.info(
            f"Saved QA correlation scores to {out_file}"
        )

        return ["correlation_scores"]