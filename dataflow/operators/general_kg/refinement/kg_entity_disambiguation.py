from dataflow.prompts.core_kg.rel_triple_refinement import KGEntityDisambiguationPrompt
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

@prompt_restrict(KGEntityDisambiguationPrompt)
@OPERATOR_REGISTRY.register()
class KGEntityDisambiguation(OperatorABC):
    r"""Processor for disambiguating entities in text using LLM-based prompts.

    This processor resolves ambiguous entity mentions in text and outputs
    disambiguated canonical forms.
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en",
        prompt_template: Union[KGEntityDisambiguationPrompt, DIYPromptABC] = None,
        num_q: int = 5
    ):
        """Initialize the entity disambiguation processor.

        Args:
            llm_serving: LLM interface for entity disambiguation.
            seed: Random seed for reproducibility.
            lang: Language setting.
            prompt_template: Optional custom prompt template.
            num_q: Number of entities to process at once.
        """
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.num_q = num_q
        self.logger = get_logger()

        if prompt_template:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = KGEntityDisambiguationPrompt(lang=self.lang)

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
                "KGEntityDisambiguation 用于对文本中的歧义实体进行消歧处理。",
                "结合原始文本上下文和待消歧实体列表，通过 LLM 输出每个实体的标准化形式。",
                "输入列 raw_chunk 为原始文本，entity 为待消歧实体列表；输出列 disambiguated_entity 为消歧后的规范化实体字符串列表。"
            )
        else:
            return (
                "KGEntityDisambiguation resolves ambiguous entity mentions in text.",
                "Combines raw text context with a list of entity candidates and uses an LLM to output the canonical form of each entity.",
                "Takes raw_chunk (str) and entity (List[str]) as input columns and outputs disambiguated_entity (List[str] of canonical entity forms)."
            )

    def process_batch(
        self, texts: List[str], entity_list: List[str], sources: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Disambiguate entities in a batch of text.

        Args:
            texts: List of text chunks.
            entity_list: List of entity lists corresponding to each text.
            sources: Optional list of source identifiers.

        Returns:
            List of dicts with source text, original entity, and disambiguated entity.
        """
        if sources is None:
            sources = ["default_source"] * len(texts)
        elif len(sources) != len(texts):
            raise ValueError("Length of sources must match length of texts")

        results = []

        for text, entities, source in tqdm(zip(texts, entity_list, sources), desc="Disambiguating entities", total=len(texts)):
            preprocessed_text = self._preprocess_text(text)

            user_inputs = [self.prompt_template.build_prompt(preprocessed_text, entities)]
            sys_prompt = self.prompt_template.build_system_prompt()

            responses = self.llm_serving.generate_from_input(user_inputs=user_inputs, system_prompt=sys_prompt)
            
            # Store result
            results.append({
                "source_text": preprocessed_text,
                "entity": entities,
                "disambiguated_entity": responses[0]
            })

        return results

    def _preprocess_text(self, text: str) -> str:
        """Preprocess input text before LLM disambiguation.

        Args:
            text: Raw input text.

        Returns:
            Preprocessed text if valid, otherwise empty string.
        """
        if not isinstance(text, str):
            return ''

        text = text.strip()
        if len(text) < 10 or len(text) > 200000:
            self.logger.warning("Text failed length check.")
            return ''
        if not self._check_text_quality(text):
            self.logger.warning("Text failed quality check.")
            return ''
        return text

    def _calculate_special_char_ratio(self, text: str) -> float:
        """Calculate ratio of special characters in text."""
        chinese_ranges = [
            (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF),
            (0x2A700, 0x2B73F), (0x2B740, 0x2B81F), (0x2B820, 0x2CEAF)
        ]
        special_count = 0
        for c in text:
            is_chinese = any(start <= ord(c) <= end for start, end in chinese_ranges)
            if not (c.isalnum() or c.isspace() or is_chinese):
                special_count += 1
        return special_count / len(text) if text else 0

    def _check_text_quality(self, text: str) -> bool:
        """Check basic quality of text: sentence count and special character ratio."""
        if text.count('。') < 2 and text.count('.') < 2:
            return False
        if self._calculate_special_char_ratio(text) > 0.3:
            return False
        return True

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        """Ensure input columns exist and output column does not conflict."""
        required_keys = [self.input_key, self.input_key_meta]
        forbidden_keys = [self.output_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"Output column(s) would be overwritten: {conflict}")

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "raw_chunk",
        input_key_meta: str = "entity",
        output_key: str = "disambiguated_entity"
    ):
        """Run entity disambiguation on a stored dataframe.

        Args:
            storage: DataFlowStorage object containing dataframe.
            input_key: Column with raw text chunks.
            input_key_meta: Column with entity lists.
            output_key: Column to save disambiguated entities.

        Returns:
            List containing output_key.
        """
        self.input_key, self.input_key_meta, self.output_key = input_key, input_key_meta, output_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        texts = dataframe[self.input_key].tolist()
        entity_list = dataframe[self.input_key_meta].tolist()

        outputs = self.process_batch(texts, entity_list)
        dataframe[self.output_key] = [o["disambiguated_entity"] for o in outputs]

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [output_key]
