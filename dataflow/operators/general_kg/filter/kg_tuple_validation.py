from dataflow.prompts.core_kg.rel_triple_filter import KGRelationTupleValidityPrompt
from dataflow.prompts.core_kg.attri_triple import KGAttributeTupleValidationPrompt
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.core.prompt import prompt_restrict, DIYPromptABC

import random
import json
import re
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm


@prompt_restrict(
    KGRelationTupleValidityPrompt,
    KGAttributeTupleValidationPrompt
    )

@OPERATOR_REGISTRY.register()
class KGTupleValidity(OperatorABC):
    """
    KGTripleValidity validates knowledge graph triples using an LLM.

    The operator takes triples as input and determines whether each triple
    is semantically valid according to the prompt specification.
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en",
        merge_to_input = False,
        triple_type: str = "relation"
    ):
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.logger = get_logger()
        self.merge_to_input = merge_to_input

        self.prompt_template = (
            KGAttributeTupleValidationPrompt(lang=self.lang)
            if triple_type == "attribute"
            else KGRelationTupleValidityPrompt(lang=self.lang)
        )

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        """
        Return a human-readable description of this operator.
        """
        if lang == "zh":
            return (
                "KGTupleValidity 用于对知识图谱中的三元组进行有效性判断。",
                "该算子使用大语言模型判断每个三元组在语义上是否合理或可信，支持关系型和属性型三元组。",
                "输入列 triple（或 tuple）为三元组字符串列表，输出列 valid_triple（或 valid_tuple）为经 LLM 验证后保留的有效三元组列表。",
            )
        else:
            return (
                "KGTupleValidity validates knowledge graph triples using an LLM.",
                "Uses a large language model to assess whether each triple is semantically valid; supports both relation and attribute triple types.",
                "Takes triple (or tuple) as input column and outputs valid_triple (or valid_tuple) containing the validated triples.",
            )

    def process_batch(
        self,
        texts: List[str],
        sources: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Validate triples in batch using the LLM.
        """
        if sources is None:
            sources = ["default_source"] * len(texts)
        elif len(sources) != len(texts):
            raise ValueError("Length of sources must match length of texts")

        results = []

        for text, source in tqdm(zip(texts, sources), total=len(texts), desc="Validating triples"):
            user_inputs = [self.prompt_template.build_prompt(text)]
            system_prompt = self.prompt_template.build_system_prompt()

            responses = self.llm_serving.generate_from_input(
                user_inputs=user_inputs,
                system_prompt=system_prompt,
            )

            parsed_output = json.loads(
                re.sub(r"```json|```|\n", "", responses[0])
            )

            results.append(
                {
                    "source_text": text,
                    "valid_triple": parsed_output,
                }
            )

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

        # 设置默认输出列
        self.output_key = "valid_triple" if self.input_key == "triple" else "valid_tuple"

        # 检查输出列冲突
        if self.output_key in dataframe.columns:
            raise ValueError(
                f"Output column '{self.output_key}' already exists and would be overwritten"
            )

        self.logger.info(f"Using input column '{self.input_key}' and output column '{self.output_key}'")

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "triple",
        output_key: str = "valid_triple",
    ):
        """
        Run the triple validity check on a DataFlow dataframe.
        """
        self.input_key = input_key
        self.output_key = output_key

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)
        texts = dataframe[self.input_key].tolist()
        outputs = self.process_batch(texts)
    
        if self.merge_to_input:
            dataframe[self.input_key] = [o["valid_triple"]["valid_triple"] for o in outputs]
            storage.write(dataframe)
            return [input_key]
        else:
            dataframe[self.output_key] = [o["valid_triple"]["valid_triple"] for o in outputs]
            storage.write(dataframe)
            return [output_key]
