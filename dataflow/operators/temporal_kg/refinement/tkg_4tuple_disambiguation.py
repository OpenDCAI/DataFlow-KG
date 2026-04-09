import pandas as pd
import random
import json
import re
from typing import List, Union
from tqdm import tqdm

from dataflow import get_logger
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.core.prompt import prompt_restrict, DIYPromptABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage

from dataflow.prompts.diverse_kg.tkg import (
    TKGRelationDisambiguationPrompt,
)
from dataflow.prompts.core_kg.rel_triple_refinement import (
    KGEntityRelationTripleDisambiguationPrompt,
)
from dataflow.prompts.core_kg.attri_triple import (
    KGAttributeTripleDisambiguationPrompt,
)


@prompt_restrict(
    KGAttributeTripleDisambiguationPrompt,
    TKGRelationDisambiguationPrompt,
)
@OPERATOR_REGISTRY.register()
class TKGTupleDisambiguation(OperatorABC):
    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en",
        attribute_prompt: Union[
            KGAttributeTripleDisambiguationPrompt, DIYPromptABC
        ] = None,
        relation_prompt: Union[
            TKGRelationDisambiguationPrompt, DIYPromptABC
        ] = None,
    ):
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.logger = get_logger()

        self.attribute_prompt = (
            attribute_prompt
            if attribute_prompt is not None
            else KGAttributeTripleDisambiguationPrompt(lang=self.lang)
        )
        self.relation_prompt = (
            relation_prompt
            if relation_prompt is not None
            else TKGRelationDisambiguationPrompt(lang=self.lang)
        )

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "TKGTupleDisambiguation 用于对合并后的时序知识图谱歧义 tuple 进行自动消岐，识别并解析属性型或关系型候选结果，输出最终解析后的无歧义 tuple。",
                "输入: 数据表中需要包含一个合并结果字段，通常由 input_key 指定，默认是 merged_tuples。"
                "该字段中的每一行通常是一个字典，其中 input_key_meta 指定具体需要处理的歧义候选列表键，默认是 ambiguous。"
                "列表中的每个元素一般是一条存在歧义的 tuple 字符串，可能是属性型表达（如含有 <attribute> 和 <value> 标签），"
                "也可能是关系型表达（如含有 <rel> 标签）。"
                "算子会自动识别 tuple 类型，并分别调用对应的大语言模型提示模板完成消岐。"
                "输出: resolved。该字段通常是一个列表，列表中的每个元素表示对应歧义 tuple 的消岐结果。"
                "若某一行没有歧义候选、模型输出解析失败，或未返回有效解析结果，则该 tuple 会回退为原始输入，"
                "若整行没有可处理内容，则输出空列表。",
            )
        else:
            return (
                "TKGTupleDisambiguation is used to automatically resolve ambiguous tuples in merged temporal knowledge graph results, identifying attribute-type or relation-type candidates and producing final disambiguated tuples.",
                "Input: the dataframe must contain a merged-result field specified by input_key, which defaults to merged_tuples. "
                "Each row in this field is usually a dictionary, where input_key_meta specifies the key of the ambiguous candidate list to process, defaulting to ambiguous. "
                "Each element in that list is typically an ambiguous tuple string, which may be an attribute-style expression (e.g. containing <attribute> and <value> tags) "
                "or a relation-style expression (e.g. containing a <rel> tag). "
                "The operator automatically detects the tuple type and calls the corresponding LLM prompt template for disambiguation. "
                "Output: resolved. This field is usually a list in which each element is the resolved result for a corresponding ambiguous tuple. "
                "If a row has no ambiguous candidates, the model output cannot be parsed successfully, or no valid resolved result is returned, the operator falls back to the original tuple; "
                "if an entire row contains no processable content, an empty list is returned for that row.",
            )

    @staticmethod
    def _detect_triple_type(triple: str) -> str:
        if "<attribute>" in triple and "<value>" in triple:
            return "attribute"
        if "<rel>" in triple:
            return "relation"
        raise ValueError(f"Unknown triple format: {triple}")

    def _resolve_single(self, triple: str) -> str:
        triple_type = self._detect_triple_type(triple)

        if triple_type == "attribute":
            prompt = self.attribute_prompt
            output_key = "resolved_attribute"
        else:
            prompt = self.relation_prompt
            output_key = "resolved_relation"

        user_prompt = prompt.build_prompt(triple)
        system_prompt = prompt.build_system_prompt()

        responses = self.llm_serving.generate_from_input(
            user_inputs=[user_prompt],
            system_prompt=system_prompt,
        )

        cleaned = re.sub(r"```json|```|\n", "", responses[0])

        try:
            parsed = json.loads(cleaned)
            resolved = parsed.get(output_key, [])
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM output: {e}")
            resolved = []

        if resolved:
            return resolved[0]

        return triple

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        if self.input_key not in dataframe.columns:
            raise ValueError(f"Missing required column: {self.input_key}")
        if self.output_key in dataframe.columns:
            raise ValueError(f"Output column already exists: {self.output_key}")

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "merged_tuples",
        input_key_meta: str = "ambiguous",
        output_key: str = "resolved",
    ):
        self.input_key = input_key
        self.input_key_meta = input_key_meta
        self.output_key = output_key

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        resolved_all = []

        for triple_dict in dataframe[self.input_key].tolist():
            triple_list = triple_dict.get(self.input_key_meta, [])
            if not triple_list:
                resolved_all.append([])
                continue

            resolved = []
            for triple in tqdm(triple_list, desc="Disambiguating KG triples"):
                resolved.append(self._resolve_single(triple))

            resolved_all.append(resolved)

        dataframe[self.output_key] = resolved_all
        output_file = storage.write(dataframe)

        self.logger.info(f"KG triple disambiguation results saved to {output_file}")
        return [output_key]