from dataflow.prompts.diverse_kg.hrkg import HRKGHyperRelationExtractorPrompt
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
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
    HRKGHyperRelationExtractorPrompt
)
@OPERATOR_REGISTRY.register()
class HRKGTripleExtraction(OperatorABC):
    r"""
    A processor for extracting knowledge graph triples from text.

    This operator takes raw text and a predefined list of valid entities as input,
    and uses an LLM-based prompt to extract entity–relation–entity triples.
    The extracted triples are written back to the dataframe for downstream
    knowledge graph construction or reasoning tasks.
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en"
    ):
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.logger = get_logger()

        self.prompt_template = (
            HRKGHyperRelationExtractorPrompt(lang=self.lang)
        )

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "HRKGTripleExtraction 用于从原始文本中抽取超关系知识图谱三元组或事件型 tuple，并将非结构化文本转换为可用于图谱构建与推理的结构化结果。",
                "输入: 数据表中需要包含文本字段，通常由 input_key 指定，默认是 raw_chunk。"
                "每一行输入应是一段待抽取的原始文本字符串，算子会先对文本做基础质量检查与预处理，例如去除过短、过长或噪声较多的文本，"
                "随后调用大语言模型与预定义提示模板，从文本中抽取知识图谱 tuple。"
                "输出: tuple。输出字段通常由 output_key 指定，默认是 tuple，"
                "其值一般为一个列表，列表中的每个元素表示一条抽取得到的三元组或超关系事件结果。"
                "若输入文本为空、质量不达标，或模型输出无法正确解析为 JSON，则该行输出为空列表。",
            )
        return (
            "HRKGTripleExtraction is used to extract hyper-relational KG triples or event-like tuples from raw text and convert unstructured text into structured results for knowledge graph construction and reasoning.",
            "Input: the dataframe must contain a text field specified by input_key, which defaults to raw_chunk. "
            "Each row should contain a piece of raw text to be processed. The operator first performs basic preprocessing and text-quality checks, "
            "such as filtering out text that is too short, too long, or too noisy, and then calls an LLM with a predefined prompt template to extract KG tuples from the text. "
            "Output: tuple. The output field is specified by output_key, which defaults to tuple. "
            "Its value is usually a list, where each element represents an extracted triple or hyper-relational event-like tuple. "
            "If the input text is empty, fails the quality check, or the LLM response cannot be parsed correctly as JSON, an empty list is returned for that row.",
        )

    def process_batch(
        self,
        texts: List[str],
        sources: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        if sources is None:
            sources = ["default_source"] * len(texts)
        elif len(sources) != len(texts):
            raise ValueError("Length of sources must match length of texts")

        raw_data = [
            {
                "text": text,
                "source": source,
            }
            for text, source in zip(texts, sources)
        ]

        return self._construct_examples(raw_data)

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key]
        forbidden_keys = [self.output_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(
                f"The following column(s) already exist and would be overwritten: {conflict}"
            )

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "raw_chunk",
        output_key: str = "tuple"
    ):
        self.input_key = input_key
        self.output_key = output_key

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        texts = dataframe[self.input_key].tolist()

        outputs = self.process_batch(texts)

        dataframe[self.output_key] = [
            o.get(self.output_key, []) for o in outputs
        ]

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [output_key]

    def _construct_examples(
        self, raw_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        self.logger.info("Starting triple extraction...")
        results = []

        for data in tqdm(raw_data, desc="Extract triples"):
            processed_text = self._preprocess_text(data.get("text", ""))
            if not processed_text:
                results.append(
                    {
                        "source_text": "",
                        "tuple": [],
                    }
                )
                continue

            user_inputs = [
                self.prompt_template.build_prompt(processed_text)
            ]
            system_prompt = self.prompt_template.build_system_prompt()

            responses = self.llm_serving.generate_from_input(
                user_inputs=user_inputs,
                system_prompt=system_prompt,
            )

            triples = self._parse_llm_response(responses[0])

            results.append(
                {
                    "source_text": processed_text,
                    "tuple": triples,
                }
            )

        return results

    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        try:
            cleaned = response.strip().strip("```json").strip("```")
            return json.loads(cleaned).get("tuple", [])
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM response: {e}")
            return []

    def _preprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = text.strip()

        if len(text) < 10 or len(text) > 200000:
            return ""

        if not self._check_text_quality(text):
            return ""

        return text

    def _calculate_special_char_ratio(self, text: str) -> float:
        chinese_ranges = [
            (0x4E00, 0x9FFF),
            (0x3400, 0x4DBF),
            (0x20000, 0x2A6DF),
            (0x2A700, 0x2B73F),
            (0x2B740, 0x2B81F),
            (0x2B820, 0x2CEAF),
        ]

        special_count = 0
        for c in text:
            is_chinese = any(start <= ord(c) <= end for start, end in chinese_ranges)
            if not (c.isalnum() or c.isspace() or is_chinese):
                special_count += 1

        return special_count / len(text) if text else 0.0

    def _check_text_quality(self, text: str) -> bool:
        if text.count("。") < 2 and text.count(".") < 2:
            return False

        if self._calculate_special_char_ratio(text) > 0.3:
            return False

        return True