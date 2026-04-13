import json
from typing import List, Dict, Any

from tqdm import tqdm

from dataflow import get_logger
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.prompts.diverse_kg.hrkg import HRKGTripleConsistencyPrompt


class HRKGTripleConsistencyEvaluator(OperatorABC):
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
        self.prompt_manager = HRKGTripleConsistencyPrompt(lang)

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "HRKGTripleConsistencyEvaluator 用于评估知识图谱 tuple/triple 列表中各条三元组或事件表达的一致性评分结果。",
                "输入: 包含 tuple 字段的数据，tuple 字段可以是 Python 列表，也可以是 JSON 字符串形式的列表；列表中的每个元素通常是一条三元组或事件字符串。"
                "算子会逐条读取每一行中的 tuple 列表，并调用大语言模型结合预定义提示模板，对这些 tuple 的整体表达一致性进行评估。"
                "输出: consistency_scores。该字段通常为一个列表，表示输入 tuple 列表中各项对应的一致性评分结果；当输入为空、JSON 解析失败或模型调用异常时，输出为空列表。",
            )
        return (
            "HRKGTripleConsistencyEvaluator is used to evaluate the consistency scores of triples or tuple-like event expressions in a KG tuple list.",
            "Input: a dataset containing a tuple field, where tuple can be either a Python list or a JSON-encoded string list. "
            "Each element in the list is typically a triple string or an event-like tuple expression. "
            "The operator reads the tuple list row by row and calls an LLM with a predefined prompt template to evaluate the overall consistency of these tuples. "
            "Output: consistency_scores. This field is usually a list containing the consistency evaluation results corresponding to the input tuples; "
            "if the input is empty, JSON parsing fails, or the LLM call raises an exception, an empty list is returned.",
        )

    def _safe_parse_json(self, response: str) -> Dict[str, Any]:
        clean = response.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(clean)
        except Exception:
            return {"consistency_scores": []}

    def process_batch(self, records: List[Dict[str, Any]]):
        results = []

        for row in tqdm(records, desc="QA Consistency Eval"):
            tuples = row.get("tuples", [])

            if isinstance(tuples, str):
                try:
                    tuples = json.loads(tuples)
                except Exception:
                    tuples = []

            if not tuples:
                results.append({
                    "consistency_scores": []
                })
                continue

            try:
                system_prompt = self.prompt_manager.build_system_prompt()
                user_prompt = self.prompt_manager.build_prompt(tuples)

                response = self.llm_serving.generate_from_input(
                    user_inputs=[user_prompt],
                    system_prompt=system_prompt
                )[0]

                data = self._safe_parse_json(response)
                scores = data.get("consistency_scores", [])

                results.append({
                    "consistency_scores": scores
                })

            except Exception as e:
                self.logger.error(f"LLM Error: {e}")
                results.append({
                    "consistency_scores": []
                })

        return results

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "tuple",
        output_key: str = "consistency_scores"
    ):
        if storage is None:
            raise ValueError("Storage required.")

        df = storage.read("dataframe")

        records = []
        for _, r in df.iterrows():
            records.append({
                "tuples": r.get(input_key, [])
            })

        outputs = self.process_batch(records)

        df[output_key] = [
            o.get(output_key, [])
            for o in outputs
        ]

        out_file = storage.write(df)

        self.logger.info(
            f"Saved QA consistency scores to {out_file}"
        )

        return ["consistency_scores"]