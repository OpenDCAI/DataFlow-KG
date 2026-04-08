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
class TKGRelationTupleDialogueQAGeneration(OperatorABC):
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

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "TKGRelationTupleDialogueQAGeneration 用于基于时序知识图谱中的多跳路径生成多轮对话式问答数据，可用于时序对话问答构建、指令微调和下游评测。",
                "输入: 数据表中需要包含多跳路径字段。字段名由 k 和 input_key_meta 共同决定，默认情况下当 k=2 时读取 2_hop_paths。"
                "每一行输入通常是一个路径列表或路径表示，内容由多条时序 tuple 或路径片段组成。"
                "算子会针对每一条路径调用大语言模型和对应的对话生成提示模板，生成围绕该路径展开的多轮 dialogue。"
                "输出: multi_turn_dialogues。该字段通常是一个列表，列表中的每个元素是一个字典，包含 path 和 dialogue 两部分："
                "其中 path 表示原始路径内容，dialogue 表示生成的多轮对话 turn 列表。"
                "若某条路径无法成功生成或解析对话结果，则该路径不会被加入当前行的输出列表。",
            )
        return (
            "TKGRelationTupleDialogueQAGeneration is used to generate multi-turn dialogue-style QA data from multi-hop paths in temporal knowledge graphs for temporal dialogue QA construction, instruction tuning, and downstream evaluation.",
            "Input: the dataframe must contain a multi-hop path field. The actual field name is determined jointly by k and input_key_meta; "
            "for example, when k=2, the operator reads 2_hop_paths by default. "
            "Each row usually contains a path list or path representation composed of multiple temporal tuples or path fragments. "
            "The operator calls an LLM with the corresponding dialogue-generation prompt template for each path and generates a multi-turn dialogue grounded in that path. "
            "Output: multi_turn_dialogues. This field is usually a list, where each element is a dictionary containing path and dialogue: "
            "path stores the original path content, and dialogue stores the generated list of dialogue turns. "
            "If a path fails to produce or parse a valid dialogue result, that path will not be included in the output list for the row.",
        )

    def _validate_dataframe(self, df: pd.DataFrame):
        if self.input_key not in df.columns:
            raise ValueError(f"Missing required column: {self.input_key}")
        if self.output_key in df.columns:
            raise ValueError(f"Output column already exists: {self.output_key}")

    def _generate_dialogue_for_path(self, path_text: str) -> List[Dict[str, Any]]:
        user_inputs = [self.prompt.build_prompt(path_text)]
        sys_prompt = self.prompt.build_system_prompt()

        responses = self.llm_serving.generate_from_input(
            user_inputs=user_inputs,
            system_prompt=sys_prompt
        )

        raw = responses[0]
        raw = raw.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"^```\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        response = json.loads(raw)
        dialogue = response["dialogue"]["turns"]

        return dialogue

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