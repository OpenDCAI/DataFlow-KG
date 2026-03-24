from dataflow.prompts.diverse_kg.hrkg import (
    HRKGRelationTripleSubgraphNumericQAPrompt,
    HRKGRelationTripleSubgraphSetQAPrompt,
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
    HRKGRelationTripleSubgraphNumericQAPrompt,
    HRKGRelationTripleSubgraphSetQAPrompt
)
@OPERATOR_REGISTRY.register()
class HRKGRelationTripleSubgraphQAGeneration(OperatorABC):
    r"""Processor for generating numeric or set-based QA pairs from HRKG subgraphs."""

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en",
        qa_type: str = "num",
        num_q: int = 5
    ):
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.num_q = num_q
        self.logger = get_logger()

        if qa_type == "num":
            self.prompt_template = HRKGRelationTripleSubgraphNumericQAPrompt(
                lang=self.lang
            )
        elif qa_type == "set":
            self.prompt_template = HRKGRelationTripleSubgraphSetQAPrompt(
                lang=self.lang
            )
        else:
            raise ValueError(f"Unsupported qa_type: {qa_type}")

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return ("HRKGRelationTripleSubgraphQAGeneration 从超关系子图生成数值型或集合型问答对。",)
        return ("HRKGRelationTripleSubgraphQAGeneration generates numeric or set-based QA pairs from hyper-relational subgraphs.",)

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

        raw_data = [{"text": text, "source": source} for text, source in zip(texts, sources)]
        results = []

        self.logger.info("Starting HRKG subgraph QA generation...")

        for data in tqdm(raw_data, desc="Generating QA pairs"):
            source_text = self._format_input_text(data["text"])

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
                cleaned_responses = json.loads(
                    re.search(r"\{.*\}", responses[0], re.DOTALL).group()
                ).get("QA_pairs", [])
            except Exception:
                self.logger.warning(f"Failed to parse LLM response: {responses[0]}")
                cleaned_responses = []

            if not isinstance(cleaned_responses, list):
                cleaned_responses = []

            results.append({
                "source_text": source_text,
                "QA_pairs": cleaned_responses
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
        input_key: str = "subgraph",
        output_key: str = "QA_pairs"
    ):
        self.input_key, self.output_key = input_key, output_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        texts = dataframe[self.input_key].tolist()
        outputs = self.process_batch(texts)

        dataframe[self.output_key] = [
            o.get("QA_pairs", [])
            for o in outputs
        ]

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [output_key]
