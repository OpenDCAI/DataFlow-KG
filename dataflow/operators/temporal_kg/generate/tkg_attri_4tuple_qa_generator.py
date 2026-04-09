from dataflow.prompts.diverse_kg.tkg import (
    TKGAttributeTimePointQAGenerationPrompt,
    TKGAttributeEventOrderQAGenerationPrompt,
    TKGAttributeTimeOrderQAGenerationPrompt,
    TKGAttributeTimeIntervalQAGenerationPrompt
)
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
import random
from typing import Any, Dict, List, Optional, Union
import json
from tqdm import tqdm
import re

from dataflow.core.prompt import prompt_restrict, DIYPromptABC


@prompt_restrict(
    TKGAttributeTimePointQAGenerationPrompt,
    TKGAttributeEventOrderQAGenerationPrompt,
    TKGAttributeTimeOrderQAGenerationPrompt,
    TKGAttributeTimeIntervalQAGenerationPrompt
)
@OPERATOR_REGISTRY.register()
class TKGAttributeQAGeneration(OperatorABC):
    r"""Processor for generating temporal QA pairs from KG tuples."""

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en",
        qa_type: str = "time_interval",
        num_q: int = 5
    ):
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.num_q = num_q
        self.logger = get_logger()

        if qa_type == 'time_point':
            self.prompt_template = TKGAttributeTimePointQAGenerationPrompt(lang=self.lang)
        elif qa_type == 'event_order':
            self.prompt_template = TKGAttributeEventOrderQAGenerationPrompt(lang=self.lang)
        elif qa_type == 'time_order':
            self.prompt_template = TKGAttributeTimeOrderQAGenerationPrompt(lang=self.lang)
        elif qa_type == 'time_interval':
            self.prompt_template = TKGAttributeTimeIntervalQAGenerationPrompt(lang=self.lang)
        else:
            raise ValueError(f"Unsupported qa_type: {qa_type}")

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "TKGAttributeQAGeneration 用于从时序知识图谱的属性型子图或相关 tuple 中生成时间相关问答对，可用于时序问答数据构建、指令微调数据生成与下游评测。",
                "输入: 数据表中需要包含一个用于生成问答的数据字段，通常由 input_key 指定，默认是 subgraph。"
                "每一行输入通常是一个字符串或子图表示，内容可包含属性型时序四元组、事件顺序信息、时间顺序信息或时间区间信息。"
                "算子会根据 qa_type 选择不同的 prompt 模板：当 qa_type='time_point' 时生成时间点问答，"
                "当 qa_type='event_order' 时生成事件顺序问答，当 qa_type='time_order' 时生成时间先后问答，"
                "当 qa_type='time_interval' 时生成时间区间问答。随后调用大语言模型生成 QA_pairs。"
                "输出: QA_pairs。该字段通常为一个列表，列表中的每个元素表示一个问答对；"
                "若模型输出无法解析为合法 JSON，或未能正确提取 QA_pairs，则该行输出为空列表。",
            )
        return (
            "TKGAttributeQAGeneration is used to generate time-related QA pairs from attribute-oriented temporal KG subgraphs or related tuples for temporal QA data construction, instruction tuning data generation, and downstream evaluation.",
            "Input: the dataframe must contain a field used for QA generation, specified by input_key, which defaults to subgraph. "
            "Each row is usually a string or a subgraph representation that may contain attribute-based temporal quadruples, event ordering information, temporal ordering information, or time interval information. "
            "The operator selects different prompt templates according to qa_type: when qa_type='time_point', it generates time-point QA pairs; "
            "when qa_type='event_order', it generates event-order QA pairs; when qa_type='time_order', it generates temporal-order QA pairs; "
            "when qa_type='time_interval', it generates time-interval QA pairs. It then calls an LLM to generate QA_pairs. "
            "Output: QA_pairs. This field is usually a list in which each element represents a question-answer pair; "
            "if the LLM output cannot be parsed as valid JSON or QA_pairs cannot be extracted correctly, an empty list is returned for that row.",
        )

    def process_batch(self, texts: List[str], sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if sources is None:
            sources = ["default_source"] * len(texts)
        elif len(sources) != len(texts):
            raise ValueError("Length of sources must match length of texts")

        raw_data = [{"text": text, "source": source} for text, source in zip(texts, sources)]
        results = []

        self.logger.info("Starting KG triple QA generation...")

        for data in tqdm(raw_data, desc="Generating QA pairs"):
            user_inputs = [self.prompt_template.build_prompt(data["text"])]
            sys_prompt = self.prompt_template.build_system_prompt()

            responses = self.llm_serving.generate_from_input(user_inputs=user_inputs, system_prompt=sys_prompt)

            try:
                cleaned_responses = json.loads(re.search(r"\{.*\}", responses[0], re.DOTALL).group())["QA_pairs"]
            except Exception:
                self.logger.warning(f"Failed to parse LLM response: {responses[0]}")
                cleaned_responses = []

            results.append({
                "source_text": data["text"],
                "QA_pairs": cleaned_responses
            })

        return results

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key]
        forbidden_keys = [self.output_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten: {conflict}")

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "subgraph",
        output_key: str = "QA_pairs"
    ):
        self.input_key, self.output_key = input_key, output_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        texts = dataframe[self.input_key].tolist()
        outputs = self.process_batch(texts)

        dataframe[self.output_key] = [o["QA_pairs"] for o in outputs]
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [output_key]