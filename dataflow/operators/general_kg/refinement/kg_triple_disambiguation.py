import pandas as pd
import random
import json
import re
from typing import List, Union
from tqdm import tqdm

from dataflow import get_logger
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.core.prompt import prompt_restrict, DIYPromptABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage

from dataflow.prompts.core_kg.attri_triple import (
    KGAttributeTripleDisambiguationPrompt,
)
from dataflow.prompts.core_kg.rel_triple_refinement import (
    KGEntityRelationTripleDisambiguationPrompt,
)


@prompt_restrict(
    KGAttributeTripleDisambiguationPrompt,
    KGEntityRelationTripleDisambiguationPrompt,
)
@OPERATOR_REGISTRY.register()
class KGTripleDisambiguation(OperatorABC):
    """
    Automatically disambiguate ambiguous KG triples.

    Supported input:
    - Attribute triples: <entity> E <attribute> A <value> V1 | V2
    - Relation triples:  <subj> H <obj> R <rel> T1 | T2

    Input column:
        ambiguous: List[str]

    Output column:
        resolved: List[str]
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en",
        attribute_prompt: Union[
            KGAttributeTripleDisambiguationPrompt, DIYPromptABC
        ] = None,
        relation_prompt: Union[
            KGEntityRelationTripleDisambiguationPrompt, DIYPromptABC
        ] = None,
    ):
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.logger = get_logger()

        self.attribute_prompt = (
            attribute_prompt
            if attribute_prompt is not None
            else KGAttributeTripleDisambiguationPrompt(lang=self.lang)
        )
        self.relation_prompt = (
            relation_prompt
            if relation_prompt is not None
            else KGEntityRelationTripleDisambiguationPrompt(lang=self.lang)
        )

    # --------------------------------------------------
    # Description
    # --------------------------------------------------
    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGTripleDisambiguation 用于对歧义的属性三元组和关系三元组进行自动消歧。",
                "从 merged_triples 列的 ambiguous 字段中读取歧义三元组，通过 LLM 选择最合理的候选值。",
                "输入列 merged_triples 为合并后的三元组字典（含 ambiguous 列表），输出列 resolved 为消歧后的三元组字符串列表。",
            )
        else:
            return (
                "KGTripleDisambiguation resolves ambiguous attribute and relation triples using an LLM.",
                "Reads ambiguous triples from the 'ambiguous' field of merged_triples and selects the most plausible candidate value.",
                "Takes merged_triples (Dict with an 'ambiguous' key) as input column and outputs resolved (List[str] of disambiguated triple strings).",
            )

    # --------------------------------------------------
    # Triple Type Detection
    # --------------------------------------------------
    @staticmethod
    def _detect_triple_type(triple: str) -> str:
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

    # --------------------------------------------------
    # Single Triple Resolution
    # --------------------------------------------------
    def _resolve_single(self, triple: str) -> str:
        triple_type = self._detect_triple_type(triple)

        if triple_type == "attribute":
            prompt = self.attribute_prompt
            output_key = "resolved_attribute"
        else:
            prompt = self.relation_prompt
            output_key = "resolved_relation"

        user_prompt = prompt.build_prompt(triple)
        system_prompt = prompt.build_system_prompt()

        responses = self.llm_serving.generate_from_input(
            user_inputs=[user_prompt],
            system_prompt=system_prompt,
        )

        cleaned = re.sub(r"```json|```|\n", "", responses[0])

        try:
            parsed = json.loads(cleaned)
            resolved = parsed.get(output_key, [])
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM output: {e}")
            resolved = []

        if resolved:
            return resolved[0]

        # fallback
        return triple

    # --------------------------------------------------
    # DataFrame Validation
    # --------------------------------------------------
    def _validate_dataframe(self, dataframe: pd.DataFrame):
        if self.input_key not in dataframe.columns:
            raise ValueError(f"Missing required column: {self.input_key}")
        if self.output_key in dataframe.columns:
            raise ValueError(f"Output column already exists: {self.output_key}")

    # --------------------------------------------------
    # Run
    # --------------------------------------------------
    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "merged_triples",
        input_key_meta: str = "ambiguous",
        output_key: str = "resolved",
    ):
        self.input_key = input_key
        self.input_key_meta = input_key_meta
        self.output_key = output_key

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        resolved_all = []

        for triple_dict in dataframe[self.input_key].tolist():
            triple_list = triple_dict.get(self.input_key_meta, [])
            if not triple_list:
                resolved_all.append([])
                continue

            resolved = []
            for triple in tqdm(triple_list, desc="Disambiguating KG triples"):
                resolved.append(self._resolve_single(triple))

            resolved_all.append(resolved)

        dataframe[self.output_key] = resolved_all
        output_file = storage.write(dataframe)

        self.logger.info(f"KG triple disambiguation results saved to {output_file}")
        return [output_key]
