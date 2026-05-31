from typing import List, Tuple, Set, Dict, Optional
from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass
from pydantic import BaseModel, ValidationError
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv, find_dotenv
import os
import torch
import logging
import threading
from pathlib import Path

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
_ = load_dotenv(find_dotenv())
logger = logging.getLogger(__name__)


def _resolve_contriever_model_path():
    env_model_path = os.getenv("HF_MODEL_PATH")
    if env_model_path:
        expanded_path = Path(env_model_path).expanduser()
        if expanded_path.is_dir():
            return str(expanded_path), True

    repo_model_path = Path(__file__).resolve().parents[3] / "facebook" / "contriever"
    if repo_model_path.is_dir():
        return str(repo_model_path), True

    return "facebook/contriever", False


def _load_contriever(device):
    model_path, local_files_only = _resolve_contriever_model_path()
    load_kwargs = {}
    hf_token = os.getenv("HF_KEY")
    if hf_token:
        load_kwargs["token"] = hf_token
    if local_files_only:
        load_kwargs["local_files_only"] = True

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, **load_kwargs)
        model = AutoModel.from_pretrained(model_path, **load_kwargs).to(device)
        return tokenizer, model
    except OSError:
        if not local_files_only:
            raise

        fallback_kwargs = {}
        if hf_token:
            fallback_kwargs["token"] = hf_token
        tokenizer = AutoTokenizer.from_pretrained("facebook/contriever", **fallback_kwargs)
        model = AutoModel.from_pretrained("facebook/contriever", **fallback_kwargs).to(
            device
        )
        return tokenizer, model


def _resolve_device(device):
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        return torch.device("cpu")

    return resolved_device


class EntityAlias(BaseModel):
    _id: int
    label: str
    alias: str
    sample_id: str
    alias_text_embedding: List[float]


class PropertyAlias(BaseModel):
    _id: int
    label: str
    alias: str
    sample_id: str
    alias_text_embedding: List[float]


class Aligner:
    def __init__(self, triplets_db, device="auto"):
        self.db = triplets_db

        self.entity_aliases_collection_name = "entity_aliases"
        self.property_aliases_collection_name = "property_aliases"

        self.property_vector_index_name = "property_aliases"
        self.entities_vector_index_name = "entity_aliases"

        self.initial_triplets_collection_name = "initial_triplets"
        self.triplets_collection_name = "triplets"
        self.filtered_triplets_collection_name = "filtered_triplets"

        self.device = _resolve_device(device)
        max_embedding_workers = max(
            1, int(os.getenv("WIKONTIC_EMBEDDING_CONCURRENCY", "1"))
        )
        self._embedding_semaphore = threading.BoundedSemaphore(max_embedding_workers)
        self.tokenizer, self.model = _load_contriever(self.device)

    def get_embedding(self, text):

        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(
                ~mask[..., None].bool(), 0.0
            )
            sentence_embeddings = (
                token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            )
            return sentence_embeddings

        if not text or not isinstance(text, str):
            return None

        with self._embedding_semaphore:
            inputs = self.tokenizer(
                [text], padding=True, truncation=True, return_tensors="pt"
            )
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
        embeddings = mean_pooling(outputs[0], inputs["attention_mask"])
        return embeddings.detach().cpu().tolist()[0]

    def retrieve_similar_properties(
        self, target_relation: str, sample_id: str, k: int = 10
    ) -> List[str]:  # List of property labels
        """
        Retrieve and rank properties that match given relation.

        Args:
            target_relation: The relation to search for
            k: Number of results to return

        Returns:
            List of property labels
        """

        collection = self.db.get_collection(self.property_aliases_collection_name)
        query_embedding = self.get_embedding(target_relation)
        if query_embedding is None:
            return []

        query_k = k * 2
        max_attempts = 5  #
        attempt = 0
        unique_ranked_properties: List[str] = []

        while len(unique_ranked_properties) < k and attempt < max_attempts:

            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.property_vector_index_name,
                        "queryVector": query_embedding,
                        "path": "alias_text_embedding",
                        "numCandidates": 150,
                        "limit": query_k if query_k < 150 else 150,
                        # "filter": {
                        #                 "sample_id": {"$eq": sample_id},
                        #             },
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "label": 1,
                        # "alias": 1
                        # "score": {"$meta": "vectorSearchScore"}
                    }
                },
            ]

            similar_properties = collection.aggregate(pipeline)

            for prop in similar_properties:
                if prop["label"] not in unique_ranked_properties:
                    unique_ranked_properties.append(prop["label"])
                if len(unique_ranked_properties) == k:
                    break

            query_k *= 2
            attempt += 1

        return unique_ranked_properties

    def retrieve_similar_entity_names(
        self, entity_name: str, sample_id: Optional[str] = None, k: int = 10
    ) -> List[str]:  # List of entity labels
        """
        Retrieve and rank entities that match given entity.

        Args:
            entity_name: The entity to search for
            k: Number of results to return

        Returns:
            List of entity labels
        """

        collection = self.db.get_collection(self.entity_aliases_collection_name)
        query_embedding = self.get_embedding(entity_name)
        if query_embedding is None:
            return []

        query_k = k * 2
        max_attempts = 5  #
        attempt = 0
        unique_ranked_entities: List[str] = []

        while len(unique_ranked_entities) < k and attempt < max_attempts:

            if sample_id is not None:
                filter = {
                    "sample_id": {"$eq": sample_id},
                }
            else:
                filter = {}

            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.entities_vector_index_name,
                        "queryVector": query_embedding,
                        "path": "alias_text_embedding",
                        "numCandidates": 150,
                        "limit": query_k if query_k < 150 else 150,
                        "filter": filter,
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "label": 1,
                        # "score": {"$meta": "vectorSearchScore"}
                    }
                },
            ]

            similar_entities = collection.aggregate(pipeline)

            for entity in similar_entities:
                if entity["label"] not in unique_ranked_entities:
                    unique_ranked_entities.append(entity["label"])
                if len(unique_ranked_entities) == k:
                    break

            query_k *= 2
            attempt += 1

        return unique_ranked_entities

    def add_entity(self, entity_name, alias, sample_id):
        collection = self.db.get_collection(self.entity_aliases_collection_name)
        if not collection.find_one(
            {"label": entity_name, "alias": alias, "sample_id": sample_id}
        ):

            collection.insert_one(
                {
                    "label": entity_name,
                    "alias": alias,
                    "sample_id": sample_id,
                    "alias_text_embedding": self.get_embedding(alias),
                }
            )

    def add_property(self, property_name, alias, sample_id):
        collection = self.db.get_collection(self.property_aliases_collection_name)
        if not collection.find_one({"label": property_name, "alias": alias}):
            collection.insert_one(
                {
                    "label": property_name,
                    "alias": alias,
                    # "sample_id": sample_id,
                    "alias_text_embedding": self.get_embedding(alias),
                }
            )

    def add_triplets(self, triplets_list, sample_id):
        collection = self.db.get_collection(self.triplets_collection_name)

        operations = []
        for triple in triplets_list:
            triple["sample_id"] = sample_id
            filter_query = {
                "subject": triple["subject"],
                "relation": triple["relation"],
                "object": triple["object"],
                "sample_id": triple["sample_id"],
            }
            operations.append(
                UpdateOne(filter_query, {"$setOnInsert": triple}, upsert=True)
            )

        if operations:
            collection.bulk_write(operations)

    def add_filtered_triplets(self, triplets_list, sample_id):
        collection = self.db.get_collection(self.filtered_triplets_collection_name)

        operations = []
        for triple in triplets_list:
            triple["sample_id"] = sample_id
            filter_query = {
                "subject": triple["subject"],
                "relation": triple["relation"],
                "object": triple["object"],
                "sample_id": triple["sample_id"],
            }
            operations.append(
                UpdateOne(filter_query, {"$setOnInsert": triple}, upsert=True)
            )

        if operations:
            collection.bulk_write(operations)

    def add_initial_triplets(self, triplets_list, sample_id):
        collection = self.db.get_collection(self.initial_triplets_collection_name)
        operations = []
        for triple in triplets_list:
            triple["sample_id"] = sample_id
            filter_query = {
                "subject": triple["subject"],
                "relation": triple["relation"],
                "object": triple["object"],
                "sample_id": triple["sample_id"],
            }
            operations.append(
                UpdateOne(filter_query, {"$setOnInsert": triple}, upsert=True)
            )
        if operations:
            collection.bulk_write(operations)
