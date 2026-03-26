from dataflow.prompts.core_kg.rel_triple_generate import KGInferredTripleGenerationPrompt, KGRelationGenerationPrompt
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.core.prompt import prompt_restrict, DIYPromptABC

import random
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm
import json
import re


@prompt_restrict(
    KGInferredTripleGenerationPrompt,
    KGRelationGenerationPrompt
)
@OPERATOR_REGISTRY.register()
class KGRelationTripleInference(OperatorABC):
    """
    KGRelationInference performs triple-level reasoning
    to infer implicit relations from existing triples.

    The operator:
    - Takes observed relation triples as input
    - Infers additional plausible triples using an LLM
    - Optionally merges inferred triples back into original triples
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en",
        with_text: bool = False,
        merge_to_input: bool = False,   # ⭐ 新增参数
    ):
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.with_text = with_text
        self.merge_to_input = merge_to_input
        self.logger = get_logger()

        self.prompt_template = (
            KGRelationGenerationPrompt(lang=self.lang)
            if self.with_text
            else KGInferredTripleGenerationPrompt(lang=self.lang)
        )

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGRelationTripleInference 用于基于已有三元组推理隐含的关系三元组。",
                "通过 LLM 对输入三元组进行推理，生成新的隐含关系三元组，可选地合并回原始输入以扩展 KG 闭包。",
                "输入列 triple 为已有关系三元组字符串列表，输出列 inferred_triple 为推理得到的新三元组列表。"
            )
        else:
            return (
                "KGRelationTripleInference infers implicit triples from observed relation triples.",
                "Uses an LLM to reason over input triples and generate new implicit relation triples; optionally merges them back into the original input for KG closure expansion.",
                "Takes triple (List[str]) as input column and outputs inferred_triple (List[str] of newly inferred triple strings)."
            )

    def process_batch(
        self,
        triples: List[str],
        text: List[str],
        sources: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:

        if sources is None:
            sources = ["default_source"] * len(triples)
        elif len(sources) != len(triples):
            raise ValueError("Length of sources must match length of triples")

        results = []

        for triple in tqdm(triples, desc="Inferring triples"):
            if self.with_text:
                user_prompt = self.prompt_template.build_prompt(triple, text)
            else:
                user_prompt = self.prompt_template.build_prompt(triple)

            system_prompt = self.prompt_template.build_system_prompt()

            responses = self.llm_serving.generate_from_input(
                user_inputs=[user_prompt],
                system_prompt=system_prompt,
            )
            print(responses)

            inferred = self._parse_llm_response(responses[0])
            results.append({"inferred_triple": inferred})

        return results

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "triple",
        output_key: str = "inferred_triple",
    ):
        self.input_key = input_key
        self.output_key = output_key

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        triples = dataframe[self.input_key].tolist()
        text = []

        if self.with_text:
            self.input_key_meta = "raw_chunk"
            text = dataframe[self.input_key_meta].tolist()

        outputs = self.process_batch(triples, text)
        inferred_triples = [o[self.output_key] for o in outputs]

        # ===== 原始行为：只写 inferred_triple =====
        dataframe[self.output_key] = inferred_triples

        # ===== ⭐ 新增：是否合并回原 triple =====
        if self.merge_to_input:
            merged_triples = []
            for original, inferred in zip(triples, inferred_triples):
                # 保持顺序去重
                seen = set()
                merged = []
                for t in original + inferred:
                    if t not in seen:
                        seen.add(t)
                        merged.append(t)
                merged_triples.append(merged)

            dataframe[self.input_key] = merged_triples
            self.logger.info("Inferred triples merged back into input triples.")

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [self.output_key]

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        if self.input_key not in dataframe.columns:
            raise ValueError(f"Missing required column: {self.input_key}")
        if self.output_key in dataframe.columns:
            raise ValueError(f"Column already exists: {self.output_key}")

    def _parse_llm_response(self, response: str) -> List[Any]:
        """
        Expected LLM output format:
        {
          "inferred_triple": [...]
        }
        """
        try:
            json_str = re.search(r"\{.*\}", response, re.DOTALL).group()
            parsed = json.loads(json_str)
            return parsed.get("inferred_triple", [])
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM response: {e}")
            return []
