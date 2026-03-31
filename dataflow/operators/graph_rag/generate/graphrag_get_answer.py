import re
import pandas as pd
from typing import List, Union
from tqdm import tqdm

from dataflow import get_logger
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC


@OPERATOR_REGISTRY.register()
class KGGraphRAGGetAnswer(OperatorABC):
    """
    Generate answers from questions + KG subgraph prompts using LLM.
    
    Input columns:
        - question: str or List[str]  # 单条问题或多问题
        - subgraph_prompt: str or List[str]  # 对应问题的 KG 子图 prompt
    
    Output columns:
        - answer: str or List[str]  # 对应问题的答案
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        lang: str = "en",
    ):
        self.llm_serving = llm_serving
        self.lang = lang
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "KGGraphRAGGetAnswer 用于基于 GraphRAG 子图提示生成答案。",
                "输入: question + subgraph_prompt; 输出: answer",
            )
        return (
            "KGGraphRAGGetAnswer is used to generate answers from GraphRAG subgraph prompts.",
            "Input: question + subgraph_prompt; Output: answer",
        )

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
    # 调用 LLM 生成答案
    # --------------------------------------------------
    def _generate_answer(self, prompt: str) -> str:
        try:
            response = self.llm_serving.generate_from_input(
                user_inputs=[prompt],
                system_prompt="You are a knowledge graph reasoning assistant. Answer based ONLY on the provided facts.",
            )
            answer = response[0]
            # 清理输出，去掉多余空格、Markdown 代码块等
            answer = re.sub(r"```.*?```", "", answer, flags=re.DOTALL).strip()
            return answer
        except Exception as e:
            self.logger.warning(f"LLM generation failed: {e}")
            return ""

    # --------------------------------------------------
    # Run
    # --------------------------------------------------
    def run(
        self,
        storage: DataFlowStorage = None,
        input_keys: List[str] = ["question", "subgraph_prompt"],
        output_key: str = "answer",
    ) -> List[str]:

        if storage is None:
            raise ValueError("storage parameter cannot be None")
        
        df = storage.read("dataframe")
        self._validate_dataframe(df, input_keys, output_key)

        answers_col = []

        for q_cell, prompt_cell in tqdm(
            zip(df[input_keys[0]], df[input_keys[1]]),
            total=len(df),
            desc="Generating answers from KG-RAG prompts",
        ):
            # ------------------------
            # CASE 1: 单条问题 / 单个 prompt
            # ------------------------
            if isinstance(q_cell, str) and isinstance(prompt_cell, str):
                answer = self._generate_answer(prompt_cell)
                answers_col.append(answer)

            # ------------------------
            # CASE 2: 多问题 / 多 prompt
            # ------------------------
            elif isinstance(q_cell, list) and isinstance(prompt_cell, list):
                row_answers = []
                # 确保长度一致
                max_len = max(len(q_cell), len(prompt_cell))
                q_cell = q_cell + [""] * (max_len - len(q_cell))
                prompt_cell = prompt_cell + [""] * (max_len - len(prompt_cell))

                for p in prompt_cell:
                    if not isinstance(p, str) or not p.strip():
                        row_answers.append("")
                        continue
                    row_answers.append(self._generate_answer(p))
                answers_col.append(row_answers)

            else:
                # 不一致或类型异常
                self.logger.warning(f"Unsupported input types: {type(q_cell)}, {type(prompt_cell)}")
                answers_col.append(None)

        df[output_key] = answers_col
        output_file = storage.write(df)
        self.logger.info(f"KG-RAG answers saved to {output_file}")

        return [output_key]
