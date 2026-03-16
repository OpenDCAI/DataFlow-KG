"""
====================================
DataFlow-KG: KGQuestionDifficultyEvaluation
====================================

Author: Zhengpin Li
Affiliation: Peking University
Email: zpli@pku.edu.cn
Created: 2026-03-16
"""

import json
import re
import random
import pandas as pd
from typing import Union
from tqdm import tqdm

from dataflow import get_logger
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.core.prompt import prompt_restrict, DIYPromptABC

from dataflow.prompts.application_kg.graph_rag import (
    KGQuestionDifficultyPrompt
)


@prompt_restrict(KGQuestionDifficultyPrompt)
@OPERATOR_REGISTRY.register()
class KGRAGQuestionDifficultyEvaluation(OperatorABC):
    """
    Evaluate the difficulty of questions.

    Input:
        question: str

    Output:
        question_difficulty: str (easy | medium | hard)
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en",
        prompt_template: Union[
            KGQuestionDifficultyPrompt, DIYPromptABC
        ] = None,
    ):
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.logger = get_logger()

        self.prompt_template = (
            prompt_template
            if prompt_template is not None
            else KGQuestionDifficultyPrompt(lang=self.lang)
        )

    # --------------------------------------------------
    # Description
    # --------------------------------------------------
    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGQuestionDifficultyEvaluation：评估问题难度。",
                "输入列：question (str)",
                "输出列：question_difficulty (easy | medium | hard)",
            )
        else:
            return (
                "KGQuestionDifficultyEvaluation evaluates question difficulty.",
                "Input column: question (str)",
                "Output column: question_difficulty (easy | medium | hard)",
            )

    # --------------------------------------------------
    # LLM call
    # --------------------------------------------------
    def _evaluate(self, question: str) -> str:

        user_prompt = self.prompt_template.build_prompt(question)
        system_prompt = self.prompt_template.build_system_prompt()

        responses = self.llm_serving.generate_from_input(
            user_inputs=[user_prompt],
            system_prompt=system_prompt,
        )

        cleaned = re.sub(r"```json|```|\n", "", responses[0])

        try:
            parsed = json.loads(cleaned)
            difficulty = parsed.get("question_difficulty", "medium")
        except Exception as e:
            self.logger.warning(
                f"Failed to parse LLM output: {e}, raw={responses[0]}"
            )
            difficulty = "medium"

        return difficulty

    # --------------------------------------------------
    # DataFrame Validation
    # --------------------------------------------------
    def _validate_dataframe(self, dataframe: pd.DataFrame):

        if self.question_key not in dataframe.columns:
            raise ValueError(f"Missing required column: {self.question_key}")

        if self.output_key in dataframe.columns:
            raise ValueError(f"Output column already exists: {self.output_key}")

    # --------------------------------------------------
    # Run
    # --------------------------------------------------
    def run(
        self,
        storage: DataFlowStorage = None,
        question_key: str = "question",
        output_key: str = "question_difficulty",
    ):

        self.question_key = question_key
        self.output_key = output_key

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        difficulties = []

        for q in tqdm(
            dataframe[self.question_key],
            total=len(dataframe),
            desc="Evaluating question difficulty",
        ):

            # -----------------------------
            # CASE 1: single question
            # -----------------------------
            if isinstance(q, str):
                difficulty = self._evaluate(q)
                difficulties.append(difficulty)

            # -----------------------------
            # CASE 2: batch questions
            # -----------------------------
            elif isinstance(q, list):

                q_scores = []
                for qi in q:
                    q_scores.append(self._evaluate(qi))

                difficulties.append(q_scores)

            else:
                raise ValueError(
                    f"Incompatible type: question={type(q)}"
                )

        dataframe[self.output_key] = difficulties

        output_file = storage.write(dataframe)
        self.logger.info(f"Question difficulty results saved to {output_file}")

        return [output_key]