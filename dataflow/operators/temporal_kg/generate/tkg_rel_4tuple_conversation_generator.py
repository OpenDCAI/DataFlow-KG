# -*- coding: utf-8 -*-
"""
====================================
DataFlow-KG: KGRelationTripletDialogueQAGeneration
====================================

Author: Zhengpin Li
Affiliation: Peking University
"""

import pandas as pd
from typing import List, Dict, Any
import json
import re
from tqdm import tqdm

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow import get_logger
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.core.prompt import prompt_restrict

from dataflow.prompts.diverse_kg.tkg import (
    TKGTupleTimePathDialogueQAGenerationPrompt
)


@prompt_restrict(
    TKGTupleTimePathDialogueQAGenerationPrompt
)
@OPERATOR_REGISTRY.register()
class TKGRelationTripletDialogueQAGeneration(OperatorABC):
    """
    Generate multi-turn dialogue QA from multi-hop KG paths.
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        lang: str = "en",
        k: int = 2,
        min_turns: int = 4
    ):
        self.llm_serving = llm_serving
        self.lang = lang
        self.k = k
        self.min_turns = min(min_turns, k)
        self.logger = get_logger()

        self.prompt = TKGTupleTimePathDialogueQAGenerationPrompt(lang=self.lang)

    # -------------------------
    # DataFrame validation
    # -------------------------
    def _validate_dataframe(self, df: pd.DataFrame):
        if self.input_key not in df.columns:
            raise ValueError(f"Missing required column: {self.input_key}")
        if self.output_key in df.columns:
            raise ValueError(f"Output column already exists: {self.output_key}")

    # -------------------------
    # One path → one dialogue
    # -------------------------
    def _generate_dialogue_for_path(self, path_text: str) -> List[Dict[str, Any]]:
        user_inputs = [self.prompt.build_prompt(path_text)]
        sys_prompt = self.prompt.build_system_prompt()

        responses = self.llm_serving.generate_from_input(
            user_inputs=user_inputs,
            system_prompt=sys_prompt
        )

        raw = responses[0]
        raw = raw.strip()
        # 去掉 ```json 和 ```
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"^```\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        response = json.loads(raw)
        dialogue = response["dialogue"]["turns"]

        return dialogue

    # -------------------------
    # Run
    # -------------------------
    def run(
        self,
        storage: DataFlowStorage,
        input_key_meta: str = "hop_paths",
        output_key: str = "multi_turn_dialogues"
    ) -> List[str]:

        self.input_key_meta = input_key_meta
        self.input_key = f"{self.k}_{self.input_key_meta}"
        self.output_key = output_key

        df = storage.read("dataframe")
        self._validate_dataframe(df)

        all_dialogues = []

        paths = df[self.input_key].tolist()

        for path_list in tqdm(paths, desc="Generate multi-turn dialogue QA"):
            row_dialogues = []
            dialogue = self._generate_dialogue_for_path(path_list)
            if dialogue:
                row_dialogues.append({
                    "path": path_list,
                    "dialogue": dialogue
                })

            all_dialogues.append(row_dialogues)

        df[self.output_key] = all_dialogues
        output_file = storage.write(df)
        self.logger.info(f"Multi-turn dialogues saved to {output_file}")

        return [self.output_key]