from dataflow.prompts.diverse_kg.legalkg import LegalKGJudgementPredictionPrompt
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage, FileStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
import random
from typing import Any, Dict, List, Optional
import json
from tqdm import tqdm
import re

from dataflow.core.prompt import prompt_restrict, DIYPromptABC
from typing import Union


@prompt_restrict(
    LegalKGJudgementPredictionPrompt
)


class LegalKGJudgementPrediction(OperatorABC):
    """
    输入：
        triple（List[str]）
        case_description（str）

    输出：
        judgement（str）
        reason（List[str]）
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "zh",
        prompt_template: Union[LegalKGJudgementPredictionPrompt, DIYPromptABC] = None,
    ):
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.logger = get_logger()

        self.prompt_template = (
            prompt_template
            if prompt_template is not None
            else LegalKGJudgementPredictionPrompt(lang=self.lang)
        )

    def _safe_parse_json(self, text: str):
        """从LLM输出中提取JSON"""
        if not text:
            return None

        # 去掉 markdown 包裹
        text = re.sub(r"```json|```", "", text).strip()

        # 提取第一个 JSON 对象
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            return None

        try:
            return json.loads(json_str)
        except Exception:
            return None

    def process_batch(
        self,
        triples_list: List[List[str]],
        case_descs: List[str],
    ) -> List[Dict[str, Any]]:

        results = []

        for triples, case_desc in tqdm(
            zip(triples_list, case_descs),
            total=len(triples_list),
            desc="Predicting judgement"
        ):
            user_input = [self.prompt_template.build_prompt(triples, case_desc)]
            system_prompt = self.prompt_template.build_system_prompt()

            response = self.llm_serving.generate_from_input(
                user_inputs=user_input,
                system_prompt=system_prompt,
            )

            parsed = self._safe_parse_json(response[0])

            if parsed is None:
                # S23 修复：原代码引用未定义变量 `raw_output`，应改为 `response[0]`。
                self.logger.error(f"Parse error. Raw output:\n{response[0]}")
                judgement, reason = None, None
            else:
                judgement = parsed.get("judgement")
                reason = parsed.get("reason")

            results.append({
                "judgement": judgement,
                "reason": reason
            })

        return results

    def _validate_dataframe(self, df, input_key):
        for key in [input_key]:
            if key not in df.columns:
                raise ValueError(f"Missing column: {key}")

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "triple",
        input_key_meta: str = "张三偷了一部苹果手机",
        output_key_judgement: str = "judgement",
        output_key_reason: str = "reason",
    ):
        df = storage.read("dataframe")
        self._validate_dataframe(df, input_key)

        triples_list = df[input_key].tolist()
        case_descs = input_key_meta

        outputs = self.process_batch(triples_list, case_descs)

        df[output_key_judgement] = [o["judgement"] for o in outputs]
        df[output_key_reason] = [o["reason"] for o in outputs]

        storage.write(df)
        return [output_key_judgement, output_key_reason]