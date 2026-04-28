import json
from typing import List, Dict, Any
from tqdm import tqdm

from dataflow import get_logger
from dataflow.core import OperatorABC, LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.core.prompt import prompt_restrict
from dataflow.prompts.diverse_kg.mmkg import MMKGVisualTripleExtractionPrompt


@OPERATOR_REGISTRY.register()
@prompt_restrict(MMKGVisualTripleExtractionPrompt)
class MMKGVisualTripleExtraction(OperatorABC):

    def __init__(
        self,
        llm_serving: LLMServingABC,
        quality_threshold: int = 3,
        lang: str = "en"
    ):
        self.logger = get_logger()

        if not isinstance(llm_serving, LLMServingABC):
            raise TypeError("llm_serving must be an instance of LLMServingABC")

        self.vlm_serving = llm_serving
        self.quality_threshold = quality_threshold
        self.lang = lang

        self.visual_prompt_manager = MMKGVisualTripleExtractionPrompt(lang=lang)

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "MMKGVisualTripleExtraction 用于根据图片和候选实体抽取视觉三元组。",
                "输入: img_dict + entity; 输出: vis_triple",
            )
        return (
            "MMKGVisualTripleExtraction is used to extract visual triples from images and candidate entities.",
            "Input: img_dict + entity; Output: vis_triple",
        )

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

    def _extract_visual_triples(
        self,
        entity_candidates: List[str],
        img_dict: Dict[str, str]
    ) -> List[str]:
        if not img_dict:
            return []

        system_prompt = self.visual_prompt_manager.build_system_prompt()

        image_paths = []
        image_labels = []
        user_prompts = []

        # 使用 img_dict 的 key 作为标识
        img_keys = list(img_dict.keys())
        img_urls = list(img_dict.values())

        for img_key, img_url in zip(img_keys, img_urls):

            # LLM prompt 里使用 img_key
            prompt = self.visual_prompt_manager.build_prompt(
                img_id=img_key,
                entity_list=entity_candidates
            )

            image_paths.append([img_url])
            image_labels.append([img_key])
            user_prompts.append(prompt)

        responses = self.vlm_serving.generate_from_input_multi_images(
            list_of_image_paths=image_paths,
            list_of_image_labels=image_labels,
            system_prompt=system_prompt,
            user_prompts=user_prompts
        )

        triples = []
        candidate_map = {e.lower(): e for e in entity_candidates}

        for img_key, response in zip(img_keys, responses):
            result = self._safe_parse_json(response)
            if not result:
                continue

            quality_score = result.get("quality_score", 5)
            if quality_score < self.quality_threshold:
                continue

            predicted_entities = result.get("entity", [])
            predicted_entities = [
                candidate_map[e.lower()]
                for e in predicted_entities
                if e.lower() in candidate_map
            ]

            for entity in predicted_entities:
                triples.append(
                    f"<subj> {entity} <obj> {img_key} <rel> depicted_in "
                )

        return list(set(triples))

    def process_single_record(
        self,
        entity_candidates: List[str],
        img_dict: Dict[str, str]
    ) -> Dict[str, Any]:

        triples = self._extract_visual_triples(
            entity_candidates,
            img_dict
        )

        return {"vis_triple": triples}

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "img_dict",
        input_key_meta: str = "entity",
        output_key: str = "vis_triple"
    ) -> List[str]:

        df = storage.read("dataframe")
        self.logger.info(f"Starting Visual Triple Extraction on {len(df)} records")

        all_triples = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting visual triples"):

            img_dict = row.get(input_key, {})
            entity_list = row.get(input_key_meta, [])

            if isinstance(img_dict, str):
                try:
                    img_dict = json.loads(img_dict)
                except:
                    img_dict = {}

            if isinstance(entity_list, str):
                entity_list = [e.strip() for e in entity_list.split(",")]

            result = self.process_single_record(entity_list, img_dict)
            all_triples.append(result["vis_triple"])

        df[output_key] = all_triples
        output_file = storage.write(df)

        total_triples = sum(len(x) for x in all_triples)
        avg_triples = total_triples / len(df) if len(df) > 0 else 0

        self.logger.info(
            f"Visual Triple Extraction complete\n"
            f"Total triples: {total_triples}\n"
            f"Avg triples per record: {avg_triples:.2f}\n"
            f"Saved to: {output_file}"
        )

        return [output_key]
