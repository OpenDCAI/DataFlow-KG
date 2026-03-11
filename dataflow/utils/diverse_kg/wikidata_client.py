"""
Wikidata API Client

A client for interacting with Wikidata API to search entities, get entity IDs,
and retrieve image URLs from Wikimedia Commons.
"""

import requests
import time
from typing import List, Dict, Any, Optional
from urllib.parse import quote
from dataflow import get_logger


class WikidataClient:
    """
    Client for Wikidata API operations.
    
    Uses Wikidata SPARQL endpoint and REST API for entity search and retrieval.
    """
    
    def __init__(self, user_agent: str = "DataFlow/1.0", max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize Wikidata client.
        
        Args:
            user_agent: User agent string for API requests
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.logger = get_logger()
        self.user_agent = user_agent
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Wikidata API endpoints
        self.sparql_endpoint = "https://query.wikidata.org/sparql"
        self.rest_api_base = "https://www.wikidata.org/w/api.php"
        self.commons_api_base = "https://commons.wikimedia.org/w/api.php"
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'application/json'
        })
    
    def _make_request(self, url: str, params: Dict[str, Any] = None, method: str = "GET") -> Optional[Dict]:  # pyright: ignore[reportArgumentType]
        """
        Make HTTP request with retry logic.
        
        Args:
            url: Request URL
            params: Query parameters
            method: HTTP method (GET or POST)
            
        Returns:
            Response JSON or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                if method == "GET":
                    response = self.session.get(url, params=params, timeout=10)
                else:
                    response = self.session.post(url, json=params, timeout=10)
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    self.logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}. Retrying...")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    self.logger.error(f"Request failed after {self.max_retries} attempts: {e}")
                    return None
        
        return None
    
    def search_entities(self, query: str, limit: int = 10, language: str = "en") -> List[Dict[str, Any]]:
        """
        Search for entities in Wikidata by name.
        
        Args:
            query: Search query (entity name)
            limit: Maximum number of results
            language: Language code for labels and descriptions
            
        Returns:
            List of candidate entities with id, label, description
        """
        if not query or not query.strip():
            return []
        
        # Use Wikidata REST API for search
        params = {
            "action": "wbsearchentities",
            "search": query,
            "language": language,
            "limit": limit,
            "format": "json"
        }
        
        response = self._make_request(self.rest_api_base, params)
        
        if not response or "search" not in response:
            return []
        
        candidates = []
        for item in response.get("search", []):
            candidates.append({
                "id": item.get("id", ""),
                "label": item.get("label", ""),
                "description": item.get("description", ""),
                "aliases": item.get("aliases", [])
            })
        
        return candidates
    
    def search_id(self, query: str, language: str = "en") -> Optional[str]:
        """
        Search for entity ID by name (returns first match).
        
        Args:
            query: Search query (entity name)
            language: Language code
            
        Returns:
            Wikidata entity ID (Q-number) or None
        """
        candidates = self.search_entities(query, limit=1, language=language)
        if candidates:
            return candidates[0].get("id")
        return None
    
    def get_entity_info(self, qid: str, language: str = "en") -> Optional[Dict[str, Any]]:
        """
        Get detailed information about an entity by QID.
        
        Args:
            qid: Wikidata entity ID (e.g., "Q317521")
            language: Language code for labels and descriptions
            
        Returns:
            Entity information dictionary or None
        """
        if not qid or not qid.startswith("Q"):
            return None
        
        params = {
            "action": "wbgetentities",
            "ids": qid,
            "props": "labels|descriptions|claims",
            "languages": language,
            "format": "json"
        }
        
        response = self._make_request(self.rest_api_base, params)
        
        if not response or "entities" not in response:
            return None
        
        entity_data = response["entities"].get(qid)
        if not entity_data:
            return None
        
        return {
            "id": qid,
            "label": entity_data.get("labels", {}).get(language, {}).get("value", ""),
            "description": entity_data.get("descriptions", {}).get(language, {}).get("value", ""),
            "claims": entity_data.get("claims", {})
        }
    
    def get_image_filename(self, qid: str) -> Optional[str]:
        """
        Get image filename from Wikimedia Commons for an entity.
        
        Args:
            qid: Wikidata entity ID (e.g., "Q317521")
            
        Returns:
            Image filename on Commons or None
        """
        if not qid or not qid.startswith("Q"):
            return None
        
        # Use REST API to get image (P18 property) - more reliable than SPARQL
        params = {
            "action": "wbgetclaims",
            "entity": qid,
            "property": "P18",  # image property
            "format": "json"
        }
        
        response = self._make_request(self.rest_api_base, params)
        
        if not response or "claims" not in response:
            return None
        
        claims = response.get("claims", {})
        p18_claims = claims.get("P18", [])
        
        if not p18_claims:
            return None
        
        # Get the first image claim value
        mainsnak = p18_claims[0].get("mainsnak", {})
        if mainsnak.get("snaktype") != "value":
            return None
        
        datavalue = mainsnak.get("datavalue", {})
        if datavalue.get("type") != "string":
            return None
        
        # The value is the filename on Commons
        filename = datavalue.get("value", "")
        
        if not filename:
            return None
        
        # Filename might already be URL-encoded, but we want the raw filename
        # Format from Wikidata is usually just the filename like "Elon_Musk_Royal_Society.jpg"
        return filename
    
    def get_image_url(self, qid: str) -> Optional[str]:
        """
        Get direct image URL from Wikimedia Commons for an entity.
        
        Args:
            qid: Wikidata entity ID (e.g., "Q317521")
            
        Returns:
            Direct image URL or None
        """
        filename = self.get_image_filename(qid)
        if not filename:
            return None
        
        # Construct direct URL using Special:FilePath
        safe_filename = quote(filename.replace(" ", "_"), safe="")
        return f"https://commons.wikimedia.org/wiki/Special:FilePath/{safe_filename}"