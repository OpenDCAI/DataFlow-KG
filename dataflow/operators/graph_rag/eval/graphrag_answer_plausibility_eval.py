import json
import re
import random
import pandas as pd
from typing import List, Dict, Union
from tqdm import tqdm

from dataflow import get_logger
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.core.prompt import prompt_restrict, DIYPromptABC

from dataflow.prompts.application_kg.graph_rag import (
    KGQuestionPlausibilityPrompt
)


@prompt_restrict(KGQuestionPlausibilityPrompt)
@OPERATOR_REGISTRY.register()
class KGRAGQuestionPlausibilityEvaluation(OperatorABC):
    """
    Evaluate the plausibility of questions given question-answer pairs.

    Input:
        question: str
        answer: str

    Output:
        question_plausibility_score: float (0-1)
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en",
        prompt_template: Union[
            KGQuestionPlausibilityPrompt, DIYPromptABC
        ] = None,
    ):
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.logger = get_logger()

        self.prompt_template = (
            prompt_template
            if prompt_template is not None
            else KGQuestionPlausibilityPrompt(lang=self.lang)
        )

    # --------------------------------------------------
    # Description
    # --------------------------------------------------
    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGRAGQuestionPlausibilityEvaluation 用于评估 GraphRAG 问答对的合理性得分。",
                "输入: question + answer; 输出: question_plausibility_score",
            )
        else:
            return (
                "KGRAGQuestionPlausibilityEvaluation is used to evaluate plausibility scores for GraphRAG question-answer pairs.",
                "Input: question + answer; Output: question_plausibility_score",
            )

    # --------------------------------------------------
    # LLM call
    # --------------------------------------------------
    def _evaluate(self, question: str, answer: str) -> float:

        user_prompt = self.prompt_template.build_prompt(question, answer)
        system_prompt = self.prompt_template.build_system_prompt()

        responses = self.llm_serving.generate_from_input(
            user_inputs=[user_prompt],
            system_prompt=system_prompt,
        )

        cleaned = re.sub(r"```json|```|\n", "", responses[0])

        try:
            parsed = json.loads(cleaned)
            score = float(parsed.get("question_plausibility_score", 0.0))
        except Exception as e:
            self.logger.warning(
                f"Failed to parse LLM output: {e}, raw={responses[0]}"
            )
            score = 0.0

        return score

    # --------------------------------------------------
    # DataFrame Validation
    # --------------------------------------------------
    def _validate_dataframe(self, dataframe: pd.DataFrame):
        if self.question_key not in dataframe.columns:
            raise ValueError(f"Missing required column: {self.question_key}")

        if self.answer_key not in dataframe.columns:
            raise ValueError(f"Missing required column: {self.answer_key}")

        if self.output_key in dataframe.columns:
            raise ValueError(f"Output column already exists: {self.output_key}")

    # --------------------------------------------------
    # Run
    # --------------------------------------------------
    def run(
        self,
        storage: DataFlowStorage = None,
        question_key: str = "question",
        answer_key: str = "answer",
        output_key: str = "question_plausibility_score",
    ):

        self.question_key = question_key
        self.answer_key = answer_key
        self.output_key = output_key

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        scores = []

        for q, a in tqdm(
            zip(dataframe[self.question_key], dataframe[self.answer_key]),
            total=len(dataframe),
            desc="Evaluating question plausibility",
        ):

            # -----------------------------
            # CASE 1: 单条 QA
            # -----------------------------
            if isinstance(q, str) and isinstance(a, str):
                score = self._evaluate(q, a)
                scores.append(score)

            # -----------------------------
            # CASE 2: batch QA
            # -----------------------------
            elif isinstance(q, list) and isinstance(a, list):

                pair_scores = []
                for qi, ai in zip(q, a):
                    pair_scores.append(self._evaluate(qi, ai))

                scores.append(pair_scores)

            else:
                raise ValueError(
                    f"Incompatible types: question={type(q)}, answer={type(a)}"
                )

        dataframe[self.output_key] = scores

        output_file = storage.write(dataframe)
        self.logger.info(f"Question plausibility results saved to {output_file}")

        return [output_key]
