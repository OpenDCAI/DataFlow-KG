"""
Wikontic - Extract ontology-aware, Wikidata-aligned knowledge graphs from raw text using LLMs.
"""

from .create_triplets_db import create_triplets_database
from .create_ontological_triplets_db import create_ontological_triplets_database
from .create_wikidata_ontology_db import create_wikidata_ontology_database

from . import utils

__all__ = [
    "create_triplets_database",
    "create_ontological_triplets_database",
    "create_wikidata_ontology_database",
    "utils",
]
