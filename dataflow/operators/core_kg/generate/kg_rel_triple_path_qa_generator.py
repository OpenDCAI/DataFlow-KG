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

from dataflow.prompts.core_kg.rel_triple_generate import KGOneHopQAPathGenerationPrompt, KGTwoHopPathQAGenerationPrompt
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
    KGOneHopQAPathGenerationPrompt,
    KGTwoHopPathQAGenerationPrompt
    )
@OPERATOR_REGISTRY.register()
class KGRelationTriplePathQAGeneration(OperatorABC):
    r"""Processor to generate one-hop QA pairs from structured triples.

    This class processes text representing triples and generates QA pairs
    for each input. The generation is done via LLM or rule-based methods,
    and the pipeline includes preprocessing, LLM prompting, and output filtering.
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang="en",
        hop=1,
        num_q: int = 5
    ):
        """Initialize the processor.

        Args:
            llm_serving (LLMServingABC): LLM interface for generating QA pairs.
            seed (int): Random seed for reproducibility.
            lang (str): Language of the input/output prompts.
            prompt_template: Optional custom prompt template.
            num_q (int): Number of QA pairs requested per input.
        """
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.num_q = num_q
        self.hop = hop
        self.logger = get_logger()

        if self.hop == 1:
            self.prompt_template = KGOneHopQAPathGenerationPrompt(lang=self.lang)
        elif self.hop == 2:
            self.prompt_template = KGTwoHopPathQAGenerationPrompt(lang=self.lang)

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        """Return a description of the processor's purpose and output format.

        Args:
            lang (str): Language for the description ('en' or 'zh').

        Returns:
            tuple: (short description, processing steps, input/output example)
        """
        if lang == "zh":
            return (
                "KGRelationTripleOneHopPathQAGeneration 是一个用于从结构化三元组生成一跳问答对的处理器。",
                "处理流程包括：文本预处理 → LLM 生成 QA 对 → 输出过滤（至少 2 个 QA 对）",
                "输入格式：\ntext: <三元组文本>\n输出格式：\n{\n  'source_text': ..., \n  'QA_pairs': [<至少2条问答对>]\n}"
            )
        else:
            return (
                "KGRelationTripleOneHopPathQAGeneration generates one-hop QA pairs from structured triples.",
                "Processing steps: text preprocessing → LLM-based QA generation → output filtering (minimum 2 QA pairs)",
                "Input format:\ntext: <triple text>\nOutput format:\n{\n  'source_text': ..., \n  'QA_pairs': [<at least 2 QA pairs>]\n}"
            )

    def process_batch(self, texts: List[str], sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Process multiple texts to generate QA pairs in batch.

        Args:
            texts (List[str]): Input texts representing triples.
            sources (Optional[List[str]]): Optional source identifiers.

        Returns:
            List[Dict[str, Any]]: List of processed examples with QA pairs.
        """
        if sources is None:
            sources = ["default_source"] * len(texts)
        elif len(sources) != len(texts):
            raise ValueError("Length of sources must match length of texts")

        raw_data = [{"text": t, "source": s} for t, s in zip(texts, sources)]
        results = []

        # Iterate over each input to generate QA pairs
        for data in tqdm(raw_data, desc="Generate QA pairs"):
            triples_text = data.get('text', '')
            # Build LLM prompt
            user_inputs = [self.prompt_template.build_prompt(triples_text)]
            sys_prompt = self.prompt_template.build_system_prompt()
            # Generate responses
            responses = self.llm_serving.generate_from_input(user_inputs=user_inputs, system_prompt=sys_prompt)

            try:
                parsed = json.loads(re.search(r'\{.*\}', responses[0], re.DOTALL).group())
                qa_pairs = parsed.get("QA_pairs", [])
            except Exception as e:
                self.logger.warning(f"Failed to parse QA_pairs: {e}")
                continue

            # Enforce minimum QA pairs constraint
            if not isinstance(qa_pairs, list) or len(qa_pairs) < 2:
                self.logger.warning(f"Drop example: need >=2 QA_pairs, got {qa_pairs}")
                continue

            results.append({
                'source_text': triples_text,
                'QA_pairs': qa_pairs
            })

        return results

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        """Validate that the dataframe contains required columns and no conflicts."""
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
        input_key_meta: str = 'hop_paths',
        output_key: str = 'QA_pairs'
    ):
        """Main entry point for processing a dataframe via DataFlowStorage.

        Args:
            storage (DataFlowStorage): Storage containing input dataframe.
            input_key (str): Column key for input text.
            output_key (str): Column key to write generated QA pairs.

        Returns:
            List[str]: List containing the output_key.
        """
        self.input_key_meta, self.output_key = input_key_meta, output_key
        dataframe = storage.read("dataframe")
        if self.hop > 1:
            self.input_key = f"{self.hop}_{self.input_key_meta}"
        elif self.hop == 1:
            self.input_key = "triple"

        self._validate_dataframe(dataframe)
        texts = dataframe[self.input_key].tolist()
        outputs = self.process_batch(texts)

        dataframe[self.output_key] = [o[self.output_key] for o in outputs]
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [self.output_key]
