from dataflow.prompts.core_kg.attri_triple import KGAttributeNormalizationPrompt
from dataflow.prompts.core_kg.rel_triple_refinement import KGRelationNormalizationPrompt
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC
import random
from typing import Any, Dict, List, Optional, Union
import json
import re
from tqdm import tqdm

from dataflow.core.prompt import prompt_restrict, DIYPromptABC

@prompt_restrict(
    KGAttributeNormalizationPrompt,
    KGRelationNormalizationPrompt
    )
@OPERATOR_REGISTRY.register()
class KGTupleNormalization(OperatorABC):
    r"""Processor for normalizing and canonicalizing KG attributes."""

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en",
        attribute_prompt: Union[
            KGAttributeNormalizationPrompt, DIYPromptABC
        ] = None,
        relation_prompt: Union[
            KGAttributeNormalizationPrompt, DIYPromptABC
        ] = None,
        num_q: int = 5
    ):
        """Initialize attribute normalization processor."""
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.num_q = num_q
        self.logger = get_logger()

        self.attribute_prompt = (
            attribute_prompt
            if attribute_prompt is not None
            else KGAttributeNormalizationPrompt(lang=self.lang)
        )
        self.relation_prompt = (
            relation_prompt
            if relation_prompt is not None
            else KGRelationNormalizationPrompt(lang=self.lang)
        )

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        """Return processor description and input/output format."""
        if lang == "zh":
            return (
                "KGTupleNormalization 用于对知识图谱三元组（属性型或关系型）进行规范化处理。",
                "通过 LLM 对三元组中的属性或关系进行同义归一、多对一标准化和去重。",
                "输入列 triple 为三元组字符串列表，输出列 normalized_triple 为规范化后的三元组列表。"
            )
        else:
            return (
                "KGTupleNormalization normalizes KG triples (attribute or relation type) using an LLM.",
                "Canonicalizes synonymous attributes or relations, applies multi-to-one mapping, and deduplicates.",
                "Takes triple (List[str]) as input column and outputs normalized_triple (List[str] of normalized triple strings)."
            )

    def _detect_triple_type(self, triple: str) -> str:
        """
        Detect whether a triple is attribute or relation.

        Returns:
            "attribute" | "relation"
        """
        if "<attribute>" in triple and "<value>" in triple:
            return "attribute"
        if "<rel>" in triple:
            return "relation"
        raise ValueError(f"Unknown triple format: {triple}")

    def process_batch(
        self, texts: List[str]
    ) -> List[Dict[str, Any]]:
        """Process a batch of attributes for normalization."""
        return self._construct_examples(texts)

    # ========== Internal: construct examples using prompt ==========
    def _construct_examples(self, raw_data: List[str]) -> List[Dict[str, Any]]:
        """Normalize attributes using the prompt and LLM."""
        self.logger.info("Starting attribute normalization...")
        results = []

        triple_type = self._detect_triple_type(raw_data[0][0])

        if triple_type == "attribute":
            prompt = self.attribute_prompt
        else:
            prompt = self.relation_prompt

        for processed_text in tqdm(raw_data, desc="Normalize attributes"):
            # Build prompt
            user_inputs = [prompt.build_prompt(processed_text)]
            sys_prompt = prompt.build_system_prompt()
            responses = self.llm_serving.generate_from_input(
                user_inputs=user_inputs, system_prompt=sys_prompt
            )

            # Clean response
            cleaned_response = re.sub(r'```json|\n|```', '', responses[0])
            try:
                json_dir = json.loads(cleaned_response)
                normalized_attribute = json_dir.get('normalized_triple', '')
            except Exception:
                self.logger.warning(f"Failed to parse LLM output: {cleaned_response}")
                normalized_attribute = ''

            results.append({
                "source_text": processed_text,
                "normalized_triple": normalized_attribute
            })

        return results

    # ========== DataFrame validation ==========
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
        self.output_key = "normalized_triple" if self.input_key == "triple" else "normalized_tuple"

        # 检查输出列冲突
        if self.output_key in dataframe.columns:
            raise ValueError(
                f"Output column '{self.output_key}' already exists and would be overwritten"
            )

        self.logger.info(f"Using input column '{self.input_key}' and output column '{self.output_key}'")


    # ========== Run pipeline ==========
    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "triple",
        output_key: str = "normalized_triple",
    ):

        """Run attribute normalization on stored dataframe."""
        self.input_key, self.output_key = input_key, output_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        texts = dataframe[self.input_key].tolist()
        outputs = self.process_batch(texts)

        dataframe[self.output_key] = [o["normalized_triple"] for o in outputs]
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [output_key]
