"""
====================================
DataFlow-KG:
====================================

Author: Wanpeng Tang
Affiliation: UESTC
Email: 2023090910014@std.uestc.edu.cn
Created: 2026-02-23

License:
    MIT License
"""

import json
import requests
import pandas as pd
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from difflib import SequenceMatcher

from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.diverse_kg.wikidata_client import WikidataClient


@OPERATOR_REGISTRY.register()  # type: ignore
class MMKGEntityLink2ImgUrl(OperatorABC):
    """
    Multi-Modal Knowledge Graph Text Entity Enrichment Operator
    
    Links text entities to Wikipedia pages and retrieves representative images
    for visualizable entity types.
    
    Input: entities field (list or comma-separated string)
    Output: linked_result field (list of formatted strings)
    Format: "<entity> Name <link> wiki_url [<image> image_url]"
    """
    
    def __init__(
        self,
        user_agent: str = "DataFlow/1.0",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        wiki_lang: str = "en",
        visualizable_types: Optional[List[str]] = None
    ):
        """
        Initialize the entity linking operator.
        
        Args:
            user_agent: User agent string for API requests
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            wiki_lang: Wikipedia language code (default: "en")
            visualizable_types: List of entity types to retrieve images for
        """
        self.logger = get_logger()
        self.user_agent = user_agent
        self.wiki_lang = wiki_lang
        
        # Initialize Wikidata client for both Wikipedia linking and image retrieval
        self.wiki_client = WikidataClient(
            user_agent=user_agent,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        
        # Wikipedia API endpoint for search
        self.wiki_api_url = f"https://{wiki_lang}.wikipedia.org/w/api.php"
        
        # Visualizable entity types (for image retrieval)
        self.visualizable_types = set(visualizable_types or [
            "Person", "Organization", "Location", "Product",
            "Building", "Animal", "Plant", "Vehicle", "Artwork"
        ])

    @staticmethod
    def get_desc(lang: str = "zh"):
        """Get operator description in specified language."""
        if lang == "zh":
            return (
                "多模态知识图谱文本实体增强算子。为文本实体添加Wikipedia链接和代表性图片。\n"
                "输入格式: entities: <列表或逗号分隔的字符串>\n"
                "输出格式: linked_result: <List[str]，格式为 '<entity> Name <link> url [<image> img_url]'>"
            )
        return (
            "Multi-Modal Knowledge Graph Text Entity Enrichment Operator.\n"
            "Input format: entities: <list or comma-separated string>\n"
            "Output format: linked_result: <List[str], format '<entity> Name <link> url [<image> img_url]'>"
        )

    # ==================== Entity Parsing ====================
    
    def _parse_entities(self, entities_input: Any) -> List[str]:
        """
        Parse entities from list or comma-separated string.
        
        Args:
            entities_input: List of entities or comma-separated string
            
        Returns:
            List of entity names
        """
        if isinstance(entities_input, str):
            # Handle JSON string
            if entities_input.strip().startswith('['):
                try:
                    entities_input = json.loads(entities_input)
                except json.JSONDecodeError:
                    pass
            
            # Handle comma-separated string
            if isinstance(entities_input, str):
                return [e.strip() for e in entities_input.split(',') if e.strip()]
        
        if isinstance(entities_input, list):
            return [str(e).strip() for e in entities_input if e]
        
        return []

    # ==================== Wikipedia Linking ====================
    
    def _wiki_search(self, entity: str, limit: int = 5) -> List[str]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": entity.strip(),
            "format": "json",
            "utf8": 1,
            "srlimit": limit
        }
        
        try:
            resp = requests.get(
                self.wiki_api_url,
                params=params,
                timeout=10,
                headers={"User-Agent": self.user_agent}
            )
            resp.raise_for_status()
            results = resp.json().get("query", {}).get("search", [])
            return [r["title"] for r in results]
        except Exception as e:  # pylint: disable=broad-except
            self.logger.warning("Wikipedia search failed for '%s': %s", entity, e)
            return []
    
    def _get_wiki_page_url(self, title: str) -> str:
        """Construct Wikipedia page URL from title."""
        from urllib.parse import quote
        safe_title = quote(title.replace(" ", "_"), safe="")
        return f"https://{self.wiki_lang}.wikipedia.org/wiki/{safe_title}"
    
    def _link_to_wikipedia(self, entity_name: str) -> Optional[Dict[str, str]]:
        """
        Link a single entity to the best matching Wikipedia page.
        
        Args:
            entity_name: Entity name to link
            
        Returns:
            Dictionary with wiki_title and wiki_url, or None if not found
        """
        if not entity_name:
            return None
        
        # Search for candidates
        candidates = self._wiki_search(entity_name)
        
        if not candidates:
            return None
        
        # Select best match using fuzzy matching
        def calc_similarity(title):
            return SequenceMatcher(None, entity_name.lower(), title.lower()).ratio()
        
        best_match = max(candidates, key=calc_similarity)
        similarity = calc_similarity(best_match)
        
        # Only accept if similarity is reasonable (> 50%)
        if similarity < 0.5:
            return None
        
        return {
            "wiki_title": best_match,
            "wiki_url": self._get_wiki_page_url(best_match)
        }

    # ==================== Visualizability Check ====================
    
    def _is_visualizable(self, entity_name: str) -> bool:
        """
        Simple heuristic to check if entity is likely visualizable.
        
        Visualizable entities typically:
        - Have capitalized words (proper nouns)
        - Are multi-word names (people, places, organizations)
        
        Args:
            entity_name: Entity name to check
            
        Returns:
            True if entity is likely visualizable
        """
        if not entity_name:
            return False
        
        words = entity_name.split()
        
        # Multi-word capitalized entities (likely Person/Location/Organization)
        if len(words) >= 2 and all(w[0].isupper() for w in words if w):
            return True
        
        # Single capitalized word (likely Location or Person)
        if len(words) == 1 and words[0] and words[0][0].isupper():
            return True
        
        return False

    # ==================== Main Entity Linking ====================
    
    def _link_single_entity(self, entity_name: str) -> Optional[str]:
        """
        Link single entity to Wikipedia and optionally get image.
        
        Args:
            entity_name: Entity name to link
            
        Returns:
            Formatted string: "<entity> Name <link> url [<image> img_url]"
            or None if linking failed
        """
        if not entity_name:
            return None
        
        # Step 1: Link to Wikipedia
        wiki_result = self._link_to_wikipedia(entity_name)
        if not wiki_result:
            return None
        
        wiki_url = wiki_result["wiki_url"]
        result = f"<entity> {entity_name} <link> {wiki_url}"
        
        # Step 2: Get image if visualizable
        if self._is_visualizable(entity_name):
            # Get Wikidata ID
            qid = self.wiki_client.search_id(entity_name)
            if qid:
                # Get representative image
                image_url = self.wiki_client.get_image_url(qid)
                if image_url:
                    result += f" <image> {image_url}"
        
        return result

    
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

    # ==================== Batch Processing ====================
    
    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "entity",
        output_key: str = "linked_result"
    ):  # type: ignore[override]
        """Run entity linking on stored dataframe."""
        df = storage.read("dataframe")
        
        linked_results = []

        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Linking entities"):
            entities_input = row.get(input_key, [])
            entity_list = self._parse_entities(entities_input)
            
            # Link each entity
            linked_entities = []
            for entity in entity_list:
                linked = self._link_single_entity(entity)
                if linked:
                    linked_entities.append(linked)
            
            linked_results.append(linked_entities)
        
        df[output_key] = linked_results
        output_file = storage.write(df)
        
        self.logger.info(f"Results saved to {output_file}")
        
        return [output_key]
