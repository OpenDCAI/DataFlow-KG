# -*- coding: utf-8 -*-
"""
====================================
DataFlow-KG:
MMKG QA Generation from Subgraph
使用 MMKGSubgraphBaseQAGenerationPrompt
====================================
"""

import json
from typing import List, Dict, Any
from tqdm import tqdm

from dataflow import get_logger
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.core.prompt import prompt_restrict
from dataflow.prompts.diverse_kg.mmkg import MMKGPathBasedQAGenerationPrompt


@OPERATOR_REGISTRY.register()
@prompt_restrict(MMKGPathBasedQAGenerationPrompt)
class MMKGPathBaseQAGeneration(OperatorABC):

    def __init__(self, llm_serving: LLMServingABC, lang: str = "en", hop=2):
        self.logger = get_logger()
        if not isinstance(llm_serving, LLMServingABC):
            raise TypeError("llm_serving must be an instance of LLMServingABC")
        self.vlm_serving = llm_serving
        self.lang = lang
        self.hop = hop
        self.visual_prompt_manager = MMKGPathBasedQAGenerationPrompt(lang=lang)

    # 安全解析JSON
    def _safe_parse_json(self, response: str) -> Dict[str, Any]:
        if not response:
            return {}
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        try:
            return json.loads(text)
        except Exception:
            return {}

    # 核心方法：根据子图 + vis_triple + 图片生成问答对
    def _generate_visual_qa_pairs(
        self,
        path: List[str],
        vis_triple: List[str],
        img_dict: Dict[str, str]
    ) -> List[Dict[str, str]]:

        if not img_dict:
            return []

        system_prompt = self.visual_prompt_manager.build_system_prompt()
        image_paths = []
        image_labels = []
        user_prompts = []

        for img_key, img_url in img_dict.items():
            prompt = self.visual_prompt_manager.build_prompt(
                img_id=img_key,
                path=path,
                vis_triple=vis_triple
            )
            image_paths.append([img_url])
            image_labels.append([img_key])
            user_prompts.append(prompt)

        # 调用 VLM 生成
        responses = self.vlm_serving.generate_from_input_multi_images(
            list_of_image_paths=image_paths,
            list_of_image_labels=image_labels,
            system_prompt=system_prompt,
            user_prompts=user_prompts
        )

        qa_pairs_all = []
        for response in responses:
            result = self._safe_parse_json(response)
            if not result:
                continue
            qa_pairs = result.get("QA_pairs", [])
            qa_pairs_all.extend(qa_pairs)

        return qa_pairs_all

    # 单条记录处理
    def process_single_record(
        self,
        subgraph: List[str],
        vis_triple: List[str],
        img_dict: Dict[str, str]
    ) -> Dict[str, Any]:
        qa_pairs = self._generate_visual_qa_pairs(subgraph, vis_triple, img_dict)
        return {"QA_pairs": qa_pairs}

    # 批量处理
    def run(
        self,
        storage: DataFlowStorage,
        input_key="vis_url",       # 图片列表路径
        input_key_meta="hop_paths", # 子图
        output_key_meta="QA_pairs"
    ) -> List[str]:

        self.input_key_meta, self.output_key_meta = input_key_meta, output_key_meta
        if self.hop > 1:
            self.input_key_meta = f"{self.hop}_{self.input_key_meta}"
        elif self.hop == 1:
            self.input_key_meta = "triple"
        self.output_key = f"{self.hop}_{self.output_key_meta}"

        df = storage.read("dataframe")
        self.logger.info(f"Starting Visual QA Generation on {len(df)} records")

        all_qas = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating QA"):
            # 构建 img_dict: key 从 vis_triple 中抽取图片ID, value 从 vis_url
            img_dict = {}
            vis_url_list = row.get(input_key, [])
            vis_triple_list = row.get("vis_triple", [])

            # 提取图片ID对应 URL
            for triple, url in zip(vis_triple_list, vis_url_list):
                # triple 格式: "<subj> X <rel> <obj> img_ID"
                img_id = triple.strip().split()[-1]
                img_dict[img_id] = url

            path = row.get(self.input_key_meta, [])
            result = self.process_single_record(path, vis_triple_list, img_dict)
            all_qas.append(result["QA_pairs"])

        df[self.output_key] = all_qas
        output_file = storage.write(df)

        self.logger.info(f"Visual QA Generation complete. Saved to: {output_file}")
        return [self.output_key]