"""
====================================
DataFlow-KG: KGRelationTripleSubgraphQAGeneration
====================================

Author: Zhengpin Li
Affiliation: Peking University
Email: zpli@pku.edu.cn
Created: 2026-01-27

License:
    MIT License
"""

from dataflow.prompts.core_kg.rel_triple_generate import KGRelationTripleSubgraphNumericQAPrompt, KGRelationTripleSubgraphSetQAPrompt
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

@prompt_restrict(KGRelationTripleSubgraphNumericQAPrompt, KGRelationTripleSubgraphSetQAPrompt)
@OPERATOR_REGISTRY.register()
class KGRelationTripleSubgraphQAGeneration(OperatorABC):
    r"""Processor for generating numeric or set-based QA pairs from KG triples.

    This processor takes multiple triples from a knowledge graph and generates
    QA pairs, either numeric ('num') or set-based ('set'), using an LLM.
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en",
        qa_type: str = "num",
        num_q: int = 5
    ):
        """Initialize the processor.

        Args:
            llm_serving: LLM interface for generating QA pairs.
            seed: Random seed.
            lang: Language setting.
            qa_type: Type of QA ('num' or 'set').
            num_q: Number of QA pairs to generate.
        """
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.num_q = num_q
        self.logger = get_logger()

        if qa_type == 'num':
            self.prompt_template = KGRelationTripleSubgraphNumericQAPrompt(lang=self.lang)
        elif qa_type == 'set':
            self.prompt_template = KGRelationTripleSubgraphSetQAPrompt(lang=self.lang)
        else:
            raise ValueError(f"Unsupported qa_type: {qa_type}")

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        """Return a description of the processor.

        Args:
            lang: Language for description ('en' or 'zh').

        Returns:
            tuple: description strings and example input/output
        """
        if lang == "zh":
            return (
                "KGRelationTripleSubgraphQAGeneration 从知识图谱三元组生成多实体问答对。",
                "处理流程：输入三元组 → 通过 LLM 构建数值或集合型 QA → 输出结构化问答",
                "输入：List[str]，每个元素为三元组字符串\n输出：List[Dict]，每个元素包含原始三元组和生成的 QA 对"
            )
        else:
            return (
                "KGRelationTripleSubgraphQAGeneration generates multi-entity QA pairs from KG triples.",
                "Processing steps: input triples → generate numeric or set-based QA via LLM → return structured QA pairs",
                "Input format: List[str], each element is a triple string\nOutput format: List[Dict], each element contains 'source_text' and 'QA_pairs'"
            )

    def process_batch(self, texts: List[str], sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Process a batch of triples into QA pairs.

        Args:
            texts: List of triple strings.
            sources: Optional list of source identifiers.

        Returns:
            List of dicts containing 'source_text' and 'QA_pairs'.
        """
        if sources is None:
            sources = ["default_source"] * len(texts)
        elif len(sources) != len(texts):
            raise ValueError("Length of sources must match length of texts")

        raw_data = [{"text": text, "source": source} for text, source in zip(texts, sources)]
        results = []

        self.logger.info("Starting KG triple QA generation...")

        for data in tqdm(raw_data, desc="Generating QA pairs"):
            # Prepare prompt
            user_inputs = [self.prompt_template.build_prompt(data["text"])]
            sys_prompt = self.prompt_template.build_system_prompt()

            # Generate QA from LLM
            responses = self.llm_serving.generate_from_input(user_inputs=user_inputs, system_prompt=sys_prompt)

            # Parse QA pairs from response
            try:
                cleaned_responses = json.loads(re.search(r"\{.*\}", responses[0], re.DOTALL).group())["QA_pairs"]
            except Exception as e:
                self.logger.warning(f"Failed to parse LLM response: {responses[0]}")
                cleaned_responses = []

            results.append({
                "source_text": data["text"],
                "QA_pairs": cleaned_responses
            })

        return results

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        """Ensure required input column exists and output column does not conflict."""
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
        """Run the processor on a dataframe stored in DataFlowStorage.

        Args:
            storage: DataFlowStorage object with the dataframe.
            input_key: Column name for input triples.
            output_key: Column name to store generated QA pairs.

        Returns:
            List containing the output_key.
        """
        self.input_key, self.output_key = input_key, output_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        texts = dataframe[self.input_key].tolist()
        outputs = self.process_batch(texts)

        dataframe[self.output_key] = [o["QA_pairs"] for o in outputs]
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [output_key]