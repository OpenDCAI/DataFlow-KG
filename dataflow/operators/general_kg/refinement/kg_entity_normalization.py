from dataflow.prompts.core_kg.rel_triple_refinement import KGEntityNormalizationPrompt
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC
import random
from typing import Any, Dict, List, Optional, Union
import json

from dataflow.core.prompt import prompt_restrict, DIYPromptABC

@prompt_restrict(KGEntityNormalizationPrompt)
@OPERATOR_REGISTRY.register()
class KGEntityNormalization(OperatorABC):
    r"""Processor for canonicalizing and deduplicating entities from text chunks."""

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en",
        prompt_template: Union[KGEntityNormalizationPrompt, DIYPromptABC] = None,
        num_q: int = 5
    ):
        """Initialize entity normalization processor."""
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.num_q = num_q
        self.logger = get_logger()

        if prompt_template:
            self.prompt_template = prompt_template
        else:
            self.prompt_template = KGEntityNormalizationPrompt(lang=self.lang)

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        """Return processor description and input/output format."""
        if lang == "zh":
            return (
                "KGEntityNormalization 用于对已抽取的实体进行统一规范化处理（不负责实体抽取）。",
                "核心功能：实体同义归一、多对一标准化、一致化命名、去重。",
                "输入列 entity 为实体字符串列表，输出列 normalized_entity 为规范化后的实体字符串列表。"
            )
        else:
            return (
                "KGEntityNormalization canonicalizes and deduplicates extracted entities (does not extract entities).",
                "Core tasks: multi-to-one entity normalization, consistent canonical naming, and deduplication.",
                "Takes entity (List[str]) as input column and outputs normalized_entity (List[str] of canonical entity strings)."
            )

    def process_batch(
        self, texts: List[str]
    ) -> Union[List[Dict[str, Any]], str]:
        """Process a batch of texts to normalize entities."""
        merged_text = self._text2merged_list(texts)
        normalization_dict = self._construct_examples(merged_text)
        return normalization_dict

    # ========== Internal: construct examples using prompt ==========
    def _construct_examples(self, raw_data: Union[str, List[str]]) -> str:
        """Call the prompt to normalize entity candidates."""
        self.logger.info("Starting entity normalization using prompt...")
        user_inputs = [self.prompt_template.build_prompt(raw_data)]
        sys_prompt = self.prompt_template.build_system_prompt()
        responses = self.llm_serving.generate_from_input(user_inputs=user_inputs, system_prompt=sys_prompt)

        # Clean up response
        cleaned = responses[0].replace('```json', '').replace('```', '').strip()
        return cleaned

    # ========== Helper: merge and deduplicate entities ==========
    def _text2merged_list(self, text: Union[str, List[str]]) -> str:
        """Convert input text or list to a JSON list of unique entities."""
        if isinstance(text, list):
            merged = ','.join(text)
        elif isinstance(text, str):
            merged = text
        else:
            raise ValueError("Input must be a string or list")

        items = [e.strip().strip('"').strip("'") for e in merged.split(',') if e.strip()]
        seen = set()
        unique_items = [e for e in items if not (e in seen or seen.add(e))]

        return json.dumps(unique_items, ensure_ascii=False)

    # ========== Normalize entities in each chunk ==========
    def _normalize_chunks(self, norm_json: Union[str, Dict[str, List[str]]], texts: List[str]) -> List[Dict[str, Any]]:
        """Map variant entities to canonical names per chunk."""
        if isinstance(norm_json, str):
            norm_json = json.loads(norm_json)

        # Build variant → canonical mapping
        variant2canon = {
            v.strip(): canon
            for canon, variants in norm_json.items()
            for v in variants if isinstance(v, str) and v.strip()
        }

        results = []
        for chunk in texts:
            entities = [e.strip() for e in chunk.split(",") if e.strip()]
            normalized_entities = [
                variant2canon.get(e, e) for e in entities
            ]
            # Deduplicate
            seen = set()
            normalized_entities = [x for x in normalized_entities if not (x in seen or seen.add(x))]
            results.append({"normalized_entity": ", ".join(normalized_entities)})

        return results

    # ========== DataFrame validation ==========
    def _validate_dataframe(self, dataframe: pd.DataFrame):
        """Check required input column and prevent output column conflict."""
        required_keys = [self.input_key]
        forbidden_keys = [self.output_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"Output column(s) would be overwritten: {conflict}")

    # ========== Run pipeline ==========
    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "entity",
        output_key: str = "normalized_entity",
    ):
        """Run entity normalization on stored dataframe."""
        self.input_key, self.output_key = input_key, output_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        texts = dataframe[self.input_key].tolist()
        normalization_dict = self.process_batch(texts)
        outputs = self._normalize_chunks(normalization_dict, texts)

        dataframe[self.output_key] = [o[output_key] for o in outputs]
        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [output_key]
