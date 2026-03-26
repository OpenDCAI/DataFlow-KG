import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
import random
from typing import Any, Dict, List, Optional, Union
import json
from tqdm import tqdm
import requests
from fuzzywuzzy import fuzz
import wikipediaapi


@OPERATOR_REGISTRY.register()
class KGEntityLink2Database(OperatorABC):
    r"""Processor for linking entities to external knowledge sources (Wikipedia)."""

    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en",
        num_q: int = 5
    ):
        """Initialize the entity linking processor.

        Args:
            llm_serving: LLM interface (optional, for any validity checks if needed).
            seed: Random seed.
            lang: Language for processing.
            prompt_template: Optional custom prompt template.
            num_q: Number of entities processed per batch.
        """
        self.rng = random.Random(seed)
        self.llm_serving = llm_serving
        self.lang = lang
        self.num_q = num_q
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        """Return description and input/output format of the processor."""
        if lang == "zh":
            return (
                "KGEntityLink2Database 用于将实体链接到外部知识库（例如 Wikipedia）。",
                "对输入实体列表进行候选检索并通过模糊匹配选择最相关的 Wikipedia 页面。",
                "输入列 entity 为实体字符串列表，输出列 linked_result 为每个实体对应的 Wikipedia 链接结果列表。"
            )
        else:
            return (
                "KGEntityLink2Database links entities to external knowledge sources (e.g., Wikipedia).",
                "Retrieves Wikipedia page candidates for each entity and selects the best match via fuzzy string matching.",
                "Takes entity (List[str]) as input column and outputs linked_result (List of dicts containing entity names and their Wikipedia URLs)."
            )

    def process_batch(
        self, texts: List[str], sources: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Process a batch of entity texts to link to Wikipedia."""
        if sources is None:
            sources = ["default_source"] * len(texts)
        elif len(sources) != len(texts):
            raise ValueError("Length of sources must match length of texts")

        results = []
        for text, source in tqdm(zip(texts, sources), desc="Linking entities", total=len(texts)):
            preprocessed_text = self._preprocess_text(text)
            entities = [e.strip() for e in preprocessed_text.split(',') if e.strip()]

            linked_result = self._link_entities(entities)

            results.append({
                "entity": text,
                "linked_result": linked_result["linked_result"]
            })

        return results

    def _preprocess_text(self, text: str) -> str:
        """Basic preprocessing: strip whitespace and normalize input."""
        if not isinstance(text, str):
            return ''
        return text.strip()

    # ========== Wikipedia Search (candidate retrieval) ==========
    def _wiki_search(self, entity: str, limit: int = 5) -> List[str]:
        """Search Wikipedia API for candidate page titles."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": entity.strip(),
            "format": "json",
            "utf8": 1,
            "srlimit": limit
        }
        WIKI_SEARCH_API = "https://en.wikipedia.org/w/api.php"
        resp = requests.get(WIKI_SEARCH_API, params=params, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("query", {}).get("search", [])
        return [r["title"] for r in results]

    # ========== Entity → Wikipedia Linking ==========
    def _link_to_wikipedia(self, entity: str) -> Dict:
        """Link a single entity to the best matching Wikipedia page."""
        wiki = wikipediaapi.Wikipedia(
            language="en",
            user_agent="KGEntityLinker/1.0",
            extract_format=wikipediaapi.ExtractFormat.WIKI
        )

        candidates = []

        # Step A: Direct page or redirect
        page = wiki.page(entity)
        if page.exists():
            candidates.append({"title": page.title, "url": page.fullurl})

        # Step B: Search API if direct match fails
        if not candidates:
            titles = self._wiki_search(entity)
            for title in titles:
                candidate_page = wiki.page(title)
                if candidate_page.exists():
                    candidates.append({"title": candidate_page.title, "url": candidate_page.fullurl})

        if not candidates:
            return {"entity": entity, "link_status": "not found"}

        # Step C: Select best match based on title similarity
        best_match = max(candidates, key=lambda x: fuzz.ratio(entity.lower(), x["title"].lower()))
        return {
            "entity": entity,
            "wiki_title": best_match["title"],
            "wiki_url": best_match["url"],
            "link_status": "success"
        }

    # ========== Batch entity linking ==========
    def _link_entities(self, entities: List[str]) -> Dict:
        """Link a list of entities to Wikipedia pages."""
        linked_result = []
        for entity in entities:
            res = self._link_to_wikipedia(entity)
            if res["link_status"] == "success":
                linked_result.append(f"<entity> {res['entity']} <link> {res['wiki_url']}")
        return {"linked_result": linked_result}

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        """Ensure input column exists and output column does not conflict."""
        required_keys = [self.input_key]
        forbidden_keys = [self.output_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"Output column(s) would be overwritten: {conflict}")

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "entity",
        output_key: str = "linked_result"
    ):
        """Run entity linking on a stored dataframe."""
        self.input_key, self.output_key = input_key, output_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        texts = dataframe[self.input_key].tolist()
        outputs = self.process_batch(texts)
        dataframe[self.output_key] = [o[self.output_key] for o in outputs]

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        return [output_key]
