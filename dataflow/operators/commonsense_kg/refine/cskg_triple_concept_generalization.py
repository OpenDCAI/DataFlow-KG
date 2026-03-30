from dataflow.prompts.diverse_kg.cskg import CSKGConceptGeneralizationPrompt
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
    CSKGConceptGeneralizationPrompt
)
@OPERATOR_REGISTRY.register()
class CSKGTripleConceptGeneralization(OperatorABC):
    r"""
    A processor for performing concept generalization on commonsense knowledge graph (CSKG) triples.

    This operator takes existing structured triples as input and uses an LLM-based prompt 
    to generalize the concepts within those triples. The generalized triples are written back 
    to the dataframe for downstream knowledge expansion or reasoning tasks.
    """

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        """
        Return a short description of the operator.
        """
        if lang == "zh":
            return (
                "CSKGTripleConceptGeneralization 用于对已有的常识知识图谱（CSKG）三元组进行概念泛化。",
                "输入为已有三元组，输出为概念泛化后的新三元组（gen_triple）。"
            )
        else:
            return (
                "CSKGTripleConceptGeneralization performs concept generalization on existing CSKG triples.",
                "Input: existing triples. Output: generalized triples (gen_triple)."
            )

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en",
        num_q: int = 5
    ):
        """
        Initialize the CSKGTripleConceptGeneralization operator.

        Args:
            llm_serving: LLM serving backend used for prompt inference.
            seed: Random seed for reproducibility.
            lang: Language setting for the prompt.
            num_q: Reserved parameter for future extensions.
        """
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.num_q = num_q
        self.logger = get_logger()

        self.prompt_template = (
                CSKGConceptGeneralizationPrompt(lang=self.lang)
            )


    def process_batch(
        self,
        texts: List[str],
        sources: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of texts for triple extraction.

        Args:
            texts: List of input text chunks.
            entity_lists: List of valid entity lists aligned with texts.
            sources: Optional source identifiers.

        Returns:
            A list of extraction results.
        """
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
        input_key: str = "triple",
        output_key: str = "gen_triple"
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

    # ------------------------------------------------------------------
    # Internal helper functions (formerly ExampleConstructor)
    # ------------------------------------------------------------------

    def _construct_examples(
        self, raw_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Construct extraction results from raw inputs.
        """
        self.logger.info("Starting triple extraction...")
        results = []

        for data in tqdm(raw_data, desc="Extract triples"):
            triple = data.get("text", "")
            user_inputs = [
                self.prompt_template.build_prompt(triple)
            ]
            system_prompt = self.prompt_template.build_system_prompt()

            responses = self.llm_serving.generate_from_input(
                user_inputs=user_inputs,
                system_prompt=system_prompt,
            )

            gen_triple = self._parse_llm_response(responses[0])

            results.append(
                {
                    "triple": triple,
                    "gen_triple": gen_triple,
                }
            )

        return results

    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        try:
            cleaned = response.strip().strip("```json").strip("```")
            return json.loads(cleaned).get("gen_triple", [])
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM response: {e}")
            return []