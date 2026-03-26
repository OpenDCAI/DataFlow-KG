from dataflow.prompts.core_kg.rel_triple_generate import KGTupleTextGenerationPrompt
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


@prompt_restrict(KGTupleTextGenerationPrompt)
@OPERATOR_REGISTRY.register()
class KGTupleTextGeneration(OperatorABC):
    """
    Operator to generate a natural language paragraph from a list of triples.
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en",
    ):
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.logger = get_logger()
        self.prompt_template = KGTupleTextGenerationPrompt(lang=self.lang)

    @staticmethod
    def get_desc(lang: str = "en"):
        if lang == "zh":
            return (
                "KGTupleTextGeneration 用于将知识图谱三元组转换为自然语言描述。",
                "读取三元组列表，通过 LLM 生成对应的自然语言段落，并写入 DataFrame。",
                "输入列 triple（可通过 input_key_meta 配置）为三元组字符串列表，输出列 description（可通过 output_key_meta 配置）为对应的自然语言文本。"
            )
        else:
            return (
                "KGTupleTextGeneration converts knowledge graph triples into natural language descriptions.",
                "Reads a list of triples, generates a natural language paragraph via LLM, and writes the result to the DataFrame.",
                "Takes triple (configurable via input_key_meta) as input column and outputs description (configurable via output_key_meta) as natural language text per row."
            )

    def process_batch(self, triples_list: List[List[str]]) -> List[Dict[str, Any]]:
        results = []
        for triples in tqdm(triples_list, desc="Generate text from triples"):
            user_inputs = [self.prompt_template.build_prompt(triples)]
            sys_prompt = self.prompt_template.build_system_prompt()
            responses = self.llm_serving.generate_from_input(user_inputs=user_inputs, system_prompt=sys_prompt)
            raw = responses[0]

            # 直接把生成文本作为 description
            results.append({
                "description": raw.strip()
            })
        return results

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        # 自动选择输入列
        if hasattr(self, "input_key") and self.input_key in dataframe.columns:
            chosen_input_key = self.input_key
        elif "triple" in dataframe.columns:
            chosen_input_key = "triple"
        elif "tuple" in dataframe.columns:
            chosen_input_key = "tuple"
        else:
            raise ValueError(
                "Missing required input column: neither 'triple' nor 'tuple' found in dataframe"
            )
        self.input_key = chosen_input_key

        # 检查输出列冲突
        if self.output_key in dataframe.columns:
            raise ValueError(
                f"Output column '{self.output_key}' already exists and would be overwritten"
            )
        self.logger.info(f"Using input column '{self.input_key}' and output column '{self.output_key}'")

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key_meta: str = 'triple',
        output_key_meta: str = 'description'
    ):
        self.input_key = input_key_meta
        self.output_key = output_key_meta
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)
        triples_list = dataframe[self.input_key].tolist()
        outputs = self.process_batch(triples_list)
        dataframe[self.output_key] = [o["description"] for o in outputs]
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        return [self.output_key]