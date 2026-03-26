from dataflow.prompts.core_kg.rel_triple_eval import KGRelationStrengthScoringPrompt
import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.core.prompt import prompt_restrict, DIYPromptABC

import random
from typing import Any, Dict, List, Optional, Union
import json
from tqdm import tqdm


@prompt_restrict(
    KGRelationStrengthScoringPrompt
)
@OPERATOR_REGISTRY.register()
class KGRelationStrengthScoring(OperatorABC):
    """
    Operator for scoring the semantic strength of knowledge graph triples
    with respect to a given text.

    Each input example consists of:
    - a text segment
    - one or more triples

    The operator queries an LLM to assign strength scores to the triples
    based on how strongly they are supported by the text.
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en",
        prompt_template: Union[KGRelationStrengthScoringPrompt, DIYPromptABC] = None,
        num_q: int = 5
    ):
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.num_q = num_q
        self.logger = get_logger()

        self.prompt_template = (
            prompt_template
            if prompt_template
            else KGRelationStrengthScoringPrompt(lang=self.lang)
        )

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        """
        Description of the operator for pipeline introspection.
        """
        if lang == "zh":
            return (
                "KGRelationStrengthScoring 是一个用于评估知识图谱三元组与文本语义一致性的处理器。",
                "该处理器基于 LLM，对给定文本中三元组的支持强度进行打分。",
                "输入包括文本片段及对应的三元组，输出为三元组的强度评分结果。"
            )
        else:
            return (
                "KGRelationStrengthScoring evaluates how strongly knowledge graph triples are supported by a given text.",
                "It uses an LLM to score the semantic consistency between text content and triples.",
                "The output is a structured strength score for each input triple."
            )

    def process_batch(
        self,
        texts: List[str],
        triples: List[Any],
        sources: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of text–triple pairs and generate strength scores.
        """
        if sources is None:
            sources = ["default_source"] * len(texts)
        elif len(sources) != len(texts):
            raise ValueError("Length of sources must match length of texts")

        results = []

        for text, triple in tqdm(zip(texts, triples), total=len(texts), desc="Scoring triples"):
            processed_text = self._preprocess_text(text)
            if not processed_text:
                results.append({"triple_strength_score": None})
                continue

            user_prompt = self.prompt_template.build_prompt(processed_text, triple)
            system_prompt = self.prompt_template.build_system_prompt()

            responses = self.llm_serving.generate_from_input(
                user_inputs=[user_prompt],
                system_prompt=system_prompt
            )

            try:
                parsed = json.loads(
                    responses[0]
                    .replace("```json", "")
                    .replace("```", "")
                    .strip()
                )
                score = parsed.get("triple_strength_score")
            except Exception as e:
                self.logger.error(f"Failed to parse LLM response: {e}")
                score = None

            results.append({"triple_strength_score": score})

        return results

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        """
        Validate required and forbidden dataframe columns.
        """
        required_keys = [self.input_key, self.input_key_meta]
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
        input_key: str = "raw_chunk",
        input_key_meta: str = "triple",
        output_key: str = "triple_strength_score"
    ):
        """
        Execute the operator on a dataframe stored in DataFlowStorage.
        """
        self.input_key = input_key
        self.input_key_meta = input_key_meta
        self.output_key = output_key

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        texts = dataframe[self.input_key].tolist()
        triples = dataframe[self.input_key_meta].tolist()

        outputs = self.process_batch(texts, triples)
        dataframe[self.output_key] = [o[self.output_key] for o in outputs]

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [output_key]

    def _preprocess_text(self, text: str) -> str:
        """
        Basic text cleaning and quality control.
        """
        if not isinstance(text, str):
            return ""

        text = text.strip()
        if len(text) < 10:
            return ""

        if text.count(".") + text.count("。") < 2:
            return ""

        return text

    def _calculate_special_char_ratio(self,text):
        # 中文字符的Unicode范围（基本汉字+扩展）
        chinese_ranges = [
            (0x4E00, 0x9FFF),    # 基本汉字
            (0x3400, 0x4DBF),    # 扩展A
            (0x20000, 0x2A6DF),  # 扩展B
            (0x2A700, 0x2B73F),  # 扩展C
            (0x2B740, 0x2B81F),  # 扩展D
            (0x2B820, 0x2CEAF)   # 扩展E
        ]
        
        special_count = 0
        for c in text:
            # 检查是否为中文、字母数字或空格
            is_chinese = any(start <= ord(c) <= end for start, end in chinese_ranges)
            if not (c.isalnum() or c.isspace() or is_chinese):
                special_count += 1
        
        return special_count / len(text) if text else 0
    
    def _check_text_quality(self, text: str) -> bool:
        r"""Check the quality of input text.

        Args:
            text (str): Text to check quality for.

        Returns:
            bool: True if text passes quality checks, False otherwise.
        """
        # 1. Basic quality check
        if (text.count('。') < 2 and text.count('.') < 2):  # Must have at least 2 sentences
            return False
        
        # 2. Special character ratio check
        special_char_ratio = self._calculate_special_char_ratio(text)
        if special_char_ratio > 0.3:  # No more than 30% special characters
            return False

        return True

    def _normalize_text_key(self, text: str):
        # 定义需要删除的停用词集合（不变）
        stopwords = {
            "the","a","an","of","and","or","in","on","at",
            "to","for","with","by","as","from","into"
        }

        words = []
        # 遍历文本中所有独立单词
        for match in re.finditer(r'\b(\w+)\b', text):
            word = match.group(1)
            # 只判断【小写版单词】是否是停用词，不改变原单词的大小写
            if word.lower() not in stopwords:
                words.append(word)
            # 如果是停用词：直接跳过，相当于删除
        
        # 【关键】用正则反向替换：把文本中的停用词（独立单词）替换为空，完美保留原格式
        pattern = r'\b(' + '|'.join(stopwords) + r')\b'
        cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        # 清理替换后产生的多余空格（仅清理停用词删除后多余的空格，不改动其他空格）
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text