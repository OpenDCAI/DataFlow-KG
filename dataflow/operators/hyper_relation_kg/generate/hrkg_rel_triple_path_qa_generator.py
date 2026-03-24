from dataflow.prompts.diverse_kg.hrkg import (
    HRKGOneHopQAPathGenerationPrompt,
    HRKGTwoHopPathQAGenerationPrompt,
)
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

from dataflow.core.prompt import prompt_restrict


@prompt_restrict(
    HRKGOneHopQAPathGenerationPrompt,
    HRKGTwoHopPathQAGenerationPrompt
)
@OPERATOR_REGISTRY.register()
class HRKGRelationTriplePathQAGeneration(OperatorABC):
    r"""Processor to generate QA pairs from hyper-relational tuples or paths."""

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en",
        hop: int = 1,
        num_q: int = 5
    ):
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.num_q = num_q
        self.hop = hop
        self.logger = get_logger()

        if self.hop == 1:
            self.prompt_template = HRKGOneHopQAPathGenerationPrompt(lang=self.lang)
        elif self.hop == 2:
            self.prompt_template = HRKGTwoHopPathQAGenerationPrompt(lang=self.lang)
        else:
            raise ValueError("Only hop=1 or hop=2 is supported.")

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return ("HRKGRelationTriplePathQAGeneration 用于从超关系 tuple 或路径中生成问答对。",)
        return (
            "HRKGRelationTriplePathQAGeneration generates QA pairs from hyper-relational tuples or paths.",)

    def _format_input_text(self, item: Any) -> str:
        if isinstance(item, list):
            return "\n".join(
                str(x) for x in item
                if isinstance(x, str) and x.strip()
            )

        if isinstance(item, str):
            return item.strip()

        return ""

    def process_batch(
        self,
        texts: List[Any],
        sources: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        if sources is None:
            sources = ["default_source"] * len(texts)
        elif len(sources) != len(texts):
            raise ValueError("Length of sources must match length of texts")

        raw_data = [{"text": t, "source": s} for t, s in zip(texts, sources)]
        results = []

        for data in tqdm(raw_data, desc="Generate QA pairs"):
            source_text = self._format_input_text(data.get("text", ""))

            if not source_text:
                results.append({
                    "source_text": "",
                    "QA_pairs": []
                })
                continue

            user_inputs = [self.prompt_template.build_prompt(source_text)]
            system_prompt = self.prompt_template.build_system_prompt()
            responses = self.llm_serving.generate_from_input(
                user_inputs=user_inputs,
                system_prompt=system_prompt
            )

            try:
                parsed = json.loads(
                    re.search(r"\{.*\}", responses[0], re.DOTALL).group()
                )
                qa_pairs = parsed.get("QA_pairs", [])
            except Exception as e:
                self.logger.warning(f"Failed to parse QA_pairs: {e}")
                qa_pairs = []

            if not isinstance(qa_pairs, list) or len(qa_pairs) < 2:
                self.logger.warning(f"Drop example: need >=2 QA_pairs, got {qa_pairs}")
                qa_pairs = []

            results.append({
                "source_text": source_text,
                "QA_pairs": qa_pairs
            })

        return results

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
        input_key_meta: str = "hop_paths",
        output_key: str = "QA_pairs"
    ):
        self.input_key_meta, self.output_key = input_key_meta, output_key
        dataframe = storage.read("dataframe")

        if self.hop > 1:
            self.input_key = f"{self.hop}_{self.input_key_meta}"
        else:
            self.input_key = "tuple"

        self._validate_dataframe(dataframe)
        texts = dataframe[self.input_key].tolist()
        outputs = self.process_batch(texts)

        dataframe[self.output_key] = [
            o.get(self.output_key, [])
            for o in outputs
        ]

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [self.output_key]
