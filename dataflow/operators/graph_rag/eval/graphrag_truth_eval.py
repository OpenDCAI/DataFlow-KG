"""
====================================
DataFlow-KG: KGAnswerLLMEvaluation
====================================

Author: Zhengpin Li
Affiliation: Peking University
Email: zpli@pku.edu.cn
Created: 2026-02-02

License:
    MIT License
"""

import re
import pandas as pd
from typing import List
from tqdm import tqdm

from dataflow import get_logger
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC


@OPERATOR_REGISTRY.register()
class KGGraphRAGAnswerLLMEvaluation(OperatorABC):
    """
    Evaluate LLM-generated answers by comparing with ground-truth answers using another LLM call.
    
    Input columns:
        - answer: str or List[str]
        - truth: str or List[str]
    
    Output columns:
        - is_correct: bool or List[bool]
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        lang: str = "en",
    ):
        self.llm_serving = llm_serving
        self.lang = lang
        self.logger = get_logger()

    # --------------------------------------------------
    # 验证 DataFrame
    # --------------------------------------------------
    def _validate_dataframe(self, df: pd.DataFrame, input_keys: List[str], output_key: str):
        for key in input_keys:
            if key not in df.columns:
                raise ValueError(f"Missing required column: {key}")
        if output_key in df.columns:
            raise ValueError(f"Output column already exists: {output_key}")

    # --------------------------------------------------
    # 调用 LLM 判断答案是否正确
    # --------------------------------------------------
    def _llm_judge_correctness(self, answer: str, truth: str) -> bool:
        """
        调用 LLM 判断 answer 是否与 truth 相符
        """
        prompt = f"""
        You are a knowledge evaluator.
        Determine whether the given answer is correct based on the ground-truth answer.
        Consider the answer correct if it contains the key information from the ground-truth, 
        even if it has additional words. Answer "True" if correct, otherwise "False".

        Ground-truth answer: {truth}
        LLM-generated answer: {answer}
        """
        try:
            response = self.llm_serving.generate_from_input(
                user_inputs=[prompt],
                system_prompt="You are a strict correctness evaluator.",
            )
            result = response[0].strip().lower()
            # 简单解析 True / False
            if "true" in result:
                return True
            elif "false" in result:
                return False
            else:
                # LLM 输出不明确时返回 False 并记录
                self.logger.warning(f"Ambiguous LLM judgment: {result}")
                return False
        except Exception as e:
            self.logger.warning(f"LLM evaluation failed: {e}")
            return False

    # --------------------------------------------------
    # Run
    # --------------------------------------------------
    def run(
        self,
        storage: DataFlowStorage = None,
        input_keys: List[str] = ["answer", "truth"],
        output_key: str = "is_correct",
    ) -> List[str]:

        if storage is None:
            raise ValueError("storage parameter cannot be None")
        
        df = storage.read("dataframe")
        self._validate_dataframe(df, input_keys, output_key)

        is_correct_col = []

        for ans_cell, truth_cell in tqdm(
            zip(df[input_keys[0]], df[input_keys[1]]),
            total=len(df),
            desc="Evaluating answers with LLM",
        ):
            # ------------------------
            # CASE 1: 单条答案
            # ------------------------
            if isinstance(ans_cell, str) and isinstance(truth_cell, str):
                is_correct_col.append(self._llm_judge_correctness(ans_cell, truth_cell))

            # ------------------------
            # CASE 2: 多答案列表
            # ------------------------
            elif isinstance(ans_cell, list) and isinstance(truth_cell, list):
                max_len = max(len(ans_cell), len(truth_cell))
                ans_cell = ans_cell + [""] * (max_len - len(ans_cell))
                truth_cell = truth_cell + [""] * (max_len - len(truth_cell))
                row_eval = [
                    self._llm_judge_correctness(a, t) for a, t in zip(ans_cell, truth_cell)
                ]
                is_correct_col.append(row_eval)

            else:
                # 类型不匹配
                self.logger.warning(f"Unsupported types: {type(ans_cell)}, {type(truth_cell)}")
                is_correct_col.append(None)

        df[output_key] = is_correct_col
        output_file = storage.write(df)
        self.logger.info(f"LLM-based answer evaluation saved to {output_file}")

        return [output_key]