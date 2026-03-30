"""
====================================
DataFlow-KG: ImgDict Direct Wiki Linking
====================================

License: MIT
"""

from typing import List, Dict, Any
from tqdm import tqdm
import pandas as pd

from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.wikidata_client import WikidataClient


@OPERATOR_REGISTRY.register()
class MMKGImgDictLink2WikiSimple(OperatorABC):
    """
    Directly link each image in img_dict to a Wikidata URL.
    Output only contains img key and wikidata_url.
    """

    def __init__(
        self,
        user_agent: str = "DataFlow/1.0",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.logger = get_logger()
        self.wiki_client = WikidataClient(
            user_agent=user_agent,
            max_retries=max_retries,
            retry_delay=retry_delay
        )

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "MMKGImgDictLink2WikiSimple 用于将 img_dict 中的图片映射到 Wikidata 链接。",
                "输入: img_dict + img_entity_mapping; 输出: linked_result",
            )
        return (
            "MMKGImgDictLink2WikiSimple is used to link images in img_dict to Wikidata entities.",
            "Input: img_dict + img_entity_mapping; Output: linked_result",
        )

    def run(
        self,
        storage: DataFlowStorage,
        input_key_img: str = "img_dict",
        output_key: str = "linked_result",
        img_entity_mapping: Dict[str, str] = None
    ) -> List[str]:
        """
        Link images in img_dict to Wikidata URLs.
        Output format: [{"img": img_key, "wikidata_url": "..."}]

        :param storage: DataFlowStorage containing dataframe
        :param input_key_img: column with img_dict
        :param output_key: column to save linked result
        :param img_entity_mapping: optional dict mapping image keys to entity names
        """
        df = storage.read("dataframe")
        self.logger.info("Starting img_dict -> Wiki linking for %d records", len(df))

        all_linked = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Linking img_dict"):
            img_dict = row.get(input_key_img, {})
            linked_entities = []

            for img_key, img_path in img_dict.items():
                # Determine entity name
                if img_entity_mapping and img_key in img_entity_mapping:
                    entity_name = img_entity_mapping[img_key]
                else:
                    # Fallback: generate entity name from img_key
                    entity_name = img_key.replace("img_", "").replace("_", " ").title()

                # Search Wikidata
                candidates = self.wiki_client.search_entities(entity_name)
                if candidates:
                    best = candidates[0]  # pick first match
                    linked_entities.append({
                        "img": img_key,
                        "wikidata_url": f"https://www.wikidata.org/wiki/{best['id']}"
                    })
                else:
                    linked_entities.append({
                        "img": img_key,
                        "wikidata_url": None
                    })

            all_linked.append(linked_entities)

        df[output_key] = all_linked
        output_file = storage.write(df)
        self.logger.info("Img_dict linking finished. Results saved to %s", output_file)
        return [output_key]
