import ast
import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List

import pandas as pd
import requests
from tqdm import tqdm

from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

@OPERATOR_REGISTRY.register()
class GeoKGEntityLink2Database(OperatorABC):
    """Link geographic entities in knowledge-graph tuples to GeoNames URLs.

    The operator automatically detects whether each tuple is a *relation*
    triple (entity–relation–entity) or an *attribute* triple
    (entity–attribute–value) and extracts the correct number of entities
    accordingly.
    """

    # ========================== init ==========================

    def __init__(
        self,
        geonames_username: str = "dataflow_kg",
        max_candidates: int = 5,
        similarity_threshold: float = 0.5,
        request_timeout: int = 10,
    ):
        """
        Args:
            geonames_username: GeoNames API username (free registration).
            max_candidates: Maximum candidate results per search.
            similarity_threshold: Minimum similarity to accept a candidate.
            request_timeout: HTTP request timeout in seconds.
        """
        self.logger = get_logger()
        self.geonames_username = geonames_username
        self.max_candidates = max_candidates
        self.similarity_threshold = similarity_threshold
        self.request_timeout = request_timeout
        self._api_url = "https://secure.geonames.org/searchJSON"

    # ========================== description ==========================

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "GeoKGEntityLink2Database 将地理三元组中的实体链接到 GeoNames 知识库。",
                "输入格式: tuple (实体-关系-实体 或 实体-属性-属性值 三元组列表)\n"
                "输出格式: linked_result (<entity> Name <link> URL)",
            )
        return (
            "GeoKGEntityLink2Database links geographic entities in tuples "
            "to GeoNames.",
            "Input: tuple (list of ER or EA triples)\n"
            "Output: linked_result (<entity> Name <link> URL)",
        )

    # ========================== run ==========================

    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "tuple",
        output_key: str = "linked_result",
    ) -> List[str]:
        """Main execution entry point.

        Args:
            storage: DataFlow storage instance.
            input_key: Column name containing tuple lists.
            output_key: Column name for linked results.

        Returns:
            List of output column names produced.
        """
        self.input_key = input_key
        self.output_key = output_key

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        all_linked: List[List[str]] = []

        for tuple_cell in tqdm(
            dataframe[self.input_key], desc="Linking geo entities"
        ):
            tuples = self._normalize_tuple_cell(tuple_cell)
            entities = self._extract_entities_from_tuples(tuples)
            linked_row = self._link_entities(entities)
            all_linked.append(linked_row)

        dataframe[self.output_key] = all_linked
        output_file = storage.write(dataframe)
        self.logger.info("Results saved to %s", output_file)
        return [output_key]

    # ========================== validation ==========================

    def _validate_dataframe(self, dataframe: pd.DataFrame) -> None:
        missing = [
            k for k in [self.input_key] if k not in dataframe.columns
        ]
        conflict = [
            k for k in [self.output_key] if k in dataframe.columns
        ]
        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(
                f"Output column(s) would be overwritten: {conflict}"
            )

    # ========================== tuple normalisation ==========================

    @staticmethod
    def _normalize_tuple_cell(cell: Any) -> List[str]:
        """Convert a DataFrame cell into ``List[str]`` of tuple strings."""
        if cell is None:
            return []

        if isinstance(cell, (list, tuple)):
            return [str(t).strip() for t in cell if str(t).strip()]

        if isinstance(cell, str):
            text = cell.strip()
            if not text:
                return []
            # Try JSON list parse
            if text.startswith("["):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return [str(t).strip() for t in parsed if str(t).strip()]
                except json.JSONDecodeError:
                    # Fallback: Python list repr (e.g. from CSV dumps)
                    try:
                        parsed = ast.literal_eval(text)
                        if isinstance(parsed, (list, tuple)):
                            return [
                                str(t).strip() for t in parsed
                                if str(t).strip()
                            ]
                    except (ValueError, SyntaxError):
                        pass
            return [text]

        try:
            if pd.isna(cell):
                return []
        except (TypeError, ValueError):
            pass

        fallback = str(cell).strip()
        return [fallback] if fallback else []

    # ========================== tuple type detection ==========================

    # Relation tuple: <subj> E1 <obj> E2 <rel> R <time> T
    _RE_SUBJ_OBJ = re.compile(r"<subj>\s*(.*?)\s*<obj>")
    _RE_OBJ_REL = re.compile(r"<obj>\s*(.*?)\s*<rel>")

    # Attribute tuple: <subj> E <attribute> A <value> V <time> T
    _RE_SUBJ_ATTR = re.compile(r"<subj>\s*(.*?)\s*<attribute>")
    _RE_ATTR_VAL = re.compile(r"<attribute>\s*(.*?)\s*<value>")

    @classmethod
    def _detect_tuple_type(cls, tuple_str: str) -> str:
        """Return ``'relation'``, ``'attribute'``, or ``'unknown'``.

        Detection is based on the presence of tag markers:
          - ``<attribute>`` and ``<value>`` → attribute tuple
          - ``<obj>`` and ``<rel>``         → relation tuple
        """
        if cls._RE_ATTR_VAL.search(tuple_str):
            return "attribute"
        if cls._RE_OBJ_REL.search(tuple_str):
            return "relation"
        return "unknown"

    # ========================== entity extraction ==========================

    def _extract_entities_from_tuples(self, tuples: List[str]) -> List[str]:
        """Extract and deduplicate entities from a list of tuple strings."""
        seen: set = set()
        entities: List[str] = []

        for t in tuples:
            for ent in self._extract_entities_single(t):
                if ent not in seen:
                    seen.add(ent)
                    entities.append(ent)

        return entities

    def _extract_entities_single(self, tuple_str: str) -> List[str]:
        """Extract entities from one tuple string.

        GeoKG tuple formats:
          - Relation:  ``<subj> E1 <obj> E2 <rel> R <time> T``
            → extracts both E1 and E2
          - Attribute: ``<subj> E <attribute> A <value> V <time> T``
            → extracts only E
        """
        if not tuple_str or not tuple_str.strip():
            return []

        ttype = self._detect_tuple_type(tuple_str)

        if ttype == "attribute":
            subj_match = self._RE_SUBJ_ATTR.search(tuple_str)
            subj = subj_match.group(1).strip() if subj_match else ""
            return [subj] if subj else []

        if ttype == "relation":
            subj_match = self._RE_SUBJ_OBJ.search(tuple_str)
            obj_match = self._RE_OBJ_REL.search(tuple_str)
            subj = subj_match.group(1).strip() if subj_match else ""
            obj = obj_match.group(1).strip() if obj_match else ""
            result: List[str] = []
            if subj:
                result.append(subj)
            if obj and obj != subj:
                result.append(obj)
            return result

        # unknown format — try to extract <subj> as fallback
        subj_match = re.search(r"<subj>\s*(.*?)\s*(?:<|$)", tuple_str)
        subj = subj_match.group(1).strip() if subj_match else ""
        return [subj] if subj else []

    # ========================== GeoNames linking ==========================

    def _link_entities(self, entities: List[str]) -> List[str]:
        """Link a list of entities and return formatted result strings."""
        results: List[str] = []
        for entity in entities:
            url = self._link_single_entity(entity)
            results.append(f"<entity> {entity} <link> {url}")
        return results

    def _link_single_entity(self, entity: str) -> str:
        """Return a GeoNames URL for *entity*, or ``'NA'``."""
        candidates = self._geonames_search(entity)
        if not candidates:
            return "NA"
        return self._select_best_candidate(entity, candidates)

    def _geonames_search(self, entity: str) -> List[Dict[str, Any]]:
        """Query the GeoNames search API."""
        if not entity:
            return []

        params = {
            "q": entity,
            "maxRows": self.max_candidates,
            "username": self.geonames_username,
        }

        try:
            resp = requests.get(
                self._api_url, params=params, timeout=self.request_timeout
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.warning(
                "GeoNames search failed for '%s': %s", entity, exc
            )
            return []

        # GeoNames returns {"status": {...}} on errors
        if isinstance(payload.get("status"), dict):
            self.logger.warning(
                "GeoNames API error for '%s': %s",
                entity,
                payload["status"].get("message", "unknown"),
            )
            return []

        results = payload.get("geonames", [])
        return results if isinstance(results, list) else []

    def _select_best_candidate(
        self, entity: str, candidates: List[Dict[str, Any]]
    ) -> str:
        """Pick the best GeoNames candidate using string similarity."""
        entity_lower = entity.strip().lower()

        def _similarity(candidate: Dict[str, Any]) -> float:
            name = str(candidate.get("name", "")).strip().lower()
            toponym = str(candidate.get("toponymName", "")).strip().lower()
            return max(
                SequenceMatcher(None, entity_lower, name).ratio(),
                SequenceMatcher(None, entity_lower, toponym).ratio(),
            )

        best = max(candidates, key=_similarity)

        if _similarity(best) < self.similarity_threshold:
            # Fall back to top result from API (ranked by relevance)
            best = candidates[0]

        geoname_id = best.get("geonameId") or best.get("geonameid")
        if geoname_id is None:
            return "NA"

        return f"https://www.geonames.org/{geoname_id}"
