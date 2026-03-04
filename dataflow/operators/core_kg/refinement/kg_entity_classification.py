"""
====================================
DataFlow-KG: KGAttributeTripleExtraction
====================================

Author: Zhengpin Li
Affiliation: Peking University
Email: zpli@pku.edu.cn
Created: 2026-01-27

License:
    MIT License
"""

from dataflow.prompts.core_kg.rel_triple_refinement import KGEntityTypeClassificationPrompt
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

from dataflow.core.prompt import prompt_restrict, DIYPromptABC

@prompt_restrict(KGEntityTypeClassificationPrompt)
@OPERATOR_REGISTRY.register()
class KGEntityClassification(OperatorABC):
    r"""Processor for classifying entity types in a knowledge graph.

    This processor takes a list of entities and predicts their types using
    a LLM-based classification prompt.
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en",
        prompt_template: Union[KGEntityTypeClassificationPrompt, DIYPromptABC] = None,
        num_q: int = 5
    ):
        """Initialize the entity classification processor.

        Args:
            llm_serving: LLM interface for entity classification.
            seed: Random seed for reproducibility.
            lang: Language setting.
            prompt_template: Custom prompt template (optional).
            num_q: Number of entities to classify at once (default=5).
        """
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.num_q = num_q
        self.logger = get_logger()

        if prompt_template:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = KGEntityTypeClassificationPrompt(lang=self.lang)

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        """Return a description of the processor and expected input/output.

        Args:
            lang: Language for description ('en' or 'zh').

        Returns:
            tuple: Description strings including input/output format.
        """
        if lang == "zh":
            return (
                "KGEntityClassification 用于对已抽取的实体进行类型分类。",
                "处理流程：输入实体列表 → LLM 类型分类 → 输出分类结果",
                "输入格式: texts: [entity1, entity2, ...]\n输出格式: List[List[str]], 每个实体对应的类型列表"
            )
        else:
            return (
                "KGEntityClassification classifies types for extracted entities.",
                "Processing steps: input entity list → LLM-based type classification → output type labels",
                "Input format: texts: [entity1, entity2, ...]\nOutput format: List[List[str]], each entity's predicted type(s)"
            )

    def process_batch(
        self, texts: List[str], sources: Optional[List[str]] = None
    ) -> List[List[str]]:
        """Process a batch of entities and classify their types.

        Args:
            texts: List of entity strings.
            sources: Optional list of source identifiers (unused).

        Returns:
            List of predicted entity types, aligned with input order.
        """
        results = []
        self.logger.info("Starting entity type classification...")

        for entity in tqdm(texts, desc="Classifying entities"):
            # Build LLM prompt
            user_inputs = [self.prompt_template.build_prompt(entity)]
            sys_prompt = self.prompt_template.build_system_prompt()

            # Generate classification
            responses = self.llm_serving.generate_from_input(user_inputs=user_inputs, system_prompt=sys_prompt)

            try:
                # Parse response as a list of types
                cleaned_response = json.loads(responses[0])
                if not isinstance(cleaned_response, list):
                    cleaned_response = [cleaned_response]
            except Exception as e:
                self.logger.warning(f"Failed to parse LLM response: {responses[0]}")
                cleaned_response = []

            results.append(cleaned_response)

        return results

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        """Ensure input column exists and output column does not conflict."""
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
        input_key: str = "entity",
        output_key: str = "entity_type"
    ):
        """Run the processor on a dataframe stored in DataFlowStorage.

        Args:
            storage: DataFlowStorage object with the dataframe.
            input_key: Column name for input entities.
            output_key: Column name to store predicted types.

        Returns:
            List containing the output_key.
        """
        self.input_key, self.output_key = input_key, output_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        texts = dataframe[self.input_key].tolist()
        outputs = self.process_batch(texts)

        dataframe[self.output_key] = outputs
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [output_key]
