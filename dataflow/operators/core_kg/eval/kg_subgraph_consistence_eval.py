import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.core.prompt import prompt_restrict, DIYPromptABC
from typing import Any, Dict, List, Optional, Union
import json
import re
from tqdm import tqdm
import random

from dataflow.prompts.core_kg.rel_triple_eval import KGSubgraphConsistencyPrompt


@prompt_restrict(KGSubgraphConsistencyPrompt)
@OPERATOR_REGISTRY.register()
class KGSubgraphConsistency(OperatorABC):
    """
    Evaluate the internal semantic consistency of a KG subgraph (set of triples).

    Each input row contains a list of triples representing a subgraph:
        [
            [subject, predicate, object],
            [subject, predicate, object],
            ...
        ]

    Output is a single consistency score between 0 (fully inconsistent)
    and 1 (fully consistent).
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en",
        merge_to_input: bool = False,
        prompt_template: Union[KGSubgraphConsistencyPrompt, DIYPromptABC] = None,
    ):
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.logger = get_logger()
        self.merge_to_input = merge_to_input

        self.prompt_template = (
            prompt_template
            if prompt_template is not None
            else KGSubgraphConsistencyPrompt(lang=self.lang)
        )

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGSubgraphConsistency 用于评估知识图谱子图内部的语义一致性。",
                "该算子使用 LLM 对输入子图中的三元组集合进行一致性打分。",
                "输入列 subgraph 为子图三元组列表，输出列 consistency_score 为 LLM 评估的一致性得分（0~1）。"
            )
        else:
            return (
                "KGSubgraphConsistency evaluates the internal semantic consistency of a KG subgraph.",
                "Uses an LLM to score the coherence of a set of triples within a subgraph.",
                "Takes subgraph (List[str] of triples) as input and outputs consistency_score (float in 0~1) per subgraph."
            )

    def process_batch(
        self,
        subgraphs: List[List[List[str]]],
        sources: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of subgraphs and generate consistency scores.
        """
        if sources is None:
            sources = ["default_source"] * len(subgraphs)
        elif len(sources) != len(subgraphs):
            raise ValueError("Length of sources must match length of subgraphs")

        results = []

        for subgraph, source in tqdm(
            zip(subgraphs, sources), total=len(subgraphs), desc="Scoring subgraph consistency"
        ):
            user_inputs = [self.prompt_template.build_prompt(subgraph)]
            system_prompt = self.prompt_template.build_system_prompt()

            responses = self.llm_serving.generate_from_input(
                user_inputs=user_inputs,
                system_prompt=system_prompt,
            )

            try:
                parsed_output = json.loads(
                    re.sub(r"```json|```|\n", "", responses[0])
                )
                score = parsed_output.get("consistency_score")
            except Exception as e:
                self.logger.error(f"Failed to parse LLM response: {e}")
                score = None

            results.append(
                {
                    "subgraph": subgraph,
                    self.output_key: score,
                }
            )

        return results

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        if self.input_key not in dataframe.columns:
            raise ValueError(f"Missing required column: {self.input_key}")
        if self.output_key in dataframe.columns:
            raise ValueError(
                f"Column '{self.output_key}' already exists and would be overwritten"
            )

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "subgraph",
        output_key: str = "consistency_score",
    ):
        """
        Run subgraph consistency evaluation on a DataFrame in DataFlowStorage.
        """
        self.input_key = input_key
        self.output_key = output_key

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        subgraphs = dataframe[self.input_key].tolist()
        outputs = self.process_batch(subgraphs)

        dataframe[self.output_key] = [o[self.output_key] for o in outputs]
        storage.write(dataframe)
        return [output_key]