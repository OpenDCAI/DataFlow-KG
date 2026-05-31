from pymongo.mongo_client import MongoClient
from pymongo.operations import SearchIndexModel
import pymongo

from typing import List
from pydantic import BaseModel, ValidationError
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import json
import time
import argparse
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_mongo_client(mongo_uri):
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=30000)
    client.admin.command("ping")
    logger.info("Connection to MongoDB successful")
    return client


def ensure_collection(db, collection_name):
    if collection_name not in db.list_collection_names():
        db.create_collection(collection_name)
        logger.info(f"Created collection: {collection_name}")


def create_search_index_for_entities(
    db,
    collection_name="entity_aliases",
    embedding_field_name="alias_text_embedding",
    index_name="entity_aliases",
    dimensions=768,
):
    logger.info(
        f"Starting to create index {index_name} for {collection_name} with dimensions {dimensions}"
    )
    collection = db.get_collection(collection_name)
    vector_search_index_model = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": True,
                "fields": {
                    embedding_field_name: {
                        "dimensions": dimensions,
                        "similarity": "cosine",
                        "type": "knnVector",
                    },
                    "sample_id": {
                        # "type": "number"
                        "type": "token"
                    },
                },
            }
        },
        name=index_name,
    )

    try:
        result = collection.create_search_index(model=vector_search_index_model)
        logger.info("Creating index...")
        time.sleep(20)
        logger.info(f"New index {index_name} created successfully: {result}")
    except Exception as e:
        logger.error(f"Error creating new vector search index {index_name}: {str(e)}")


def create_search_index_for_properties(
    db,
    collection_name="property_aliases",
    embedding_field_name="alias_text_embedding",
    index_name="property_aliases",
    dimensions=768,
):
    logger.info(
        f"Starting to create index {index_name} for {collection_name} with dimensions {dimensions}"
    )
    collection = db.get_collection(collection_name)
    vector_search_index_model = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": True,
                "fields": {
                    embedding_field_name: {
                        "dimensions": dimensions,
                        "similarity": "cosine",
                        "type": "knnVector",
                    },
                },
            }
        },
        name=index_name,
    )
    try:
        result = collection.create_search_index(model=vector_search_index_model)
        logger.info("Creating index...")
        time.sleep(20)
        logger.info(f"New index {index_name} created successfully: {result}")
    except Exception as e:
        logger.error(f"Error creating new vector search index {index_name}: {str(e)}")


def create_triplets_database(
    mongo_uri: str = "mongodb://localhost:27018/?directConnection=true",
    db_name: str = "triplets_db",
    entity_aliases_collection: str = "entity_aliases",
    property_aliases_collection: str = "property_aliases",
    triplets_collection: str = "triplets",
    initial_triplets_collection: str = "initial_triplets",
    filtered_triplets_collection: str = "filtered_triplets",
    entity_aliases_index: str = "entity_aliases",
    property_aliases_index: str = "property_aliases",
    embedding_dimensions: int = 768,
    drop_collections: bool = False,
):
    """
    Create collections and indexes for the dynamic triplets database.

    Args:
        mongo_uri: MongoDB connection URI
        db_name: Name of the database to create
        entity_aliases_collection: Collection name for entity aliases
        property_aliases_collection: Collection name for property aliases
        triplets_collection: Collection name for triplets
        initial_triplets_collection: Collection name for initial triplets
        filtered_triplets_collection: Collection name for filtered triplets
        entity_aliases_index: Index name for entities
        property_aliases_index: Index name for property aliases
        embedding_dimensions: Dimensions for embeddings (default: 768)
        drop_collections: Whether to drop existing collections before creating new ones

    Returns:
        Database object
    """
    mongo_client = get_mongo_client(mongo_uri)
    db = mongo_client.get_database(db_name)

    # Drop all existing collections
    if drop_collections:
        logger.info("Dropping existing collections...")
        for collection_name in db.list_collection_names():
            logger.info(f"Dropping collection: {collection_name}")
            db.drop_collection(collection_name)
        logger.info("Successfully dropped all existing collections")

    ensure_collection(db, entity_aliases_collection)
    ensure_collection(db, property_aliases_collection)
    ensure_collection(db, triplets_collection)
    ensure_collection(db, initial_triplets_collection)
    ensure_collection(db, filtered_triplets_collection)

    logger.info("Collections created successfully")
    db.entity_aliases.create_index([("sample_id", 1)])
    db.entity_aliases.create_index([("label", 1)])

    db.property_aliases.create_index([("sample_id", 1)])
    db.property_aliases.create_index([("label", 1)])

    db.triplets.create_index([("sample_id", 1)])
    db.initial_triplets.create_index([("sample_id", 1)])
    db.filtered_triplets.create_index([("sample_id", 1)])
    logger.info("Indexes created successfully")

    create_search_index_for_entities(
        db,
        collection_name=entity_aliases_collection,
        embedding_field_name="alias_text_embedding",
        index_name=entity_aliases_index,
        dimensions=embedding_dimensions,
    )
    create_search_index_for_properties(
        db,
        collection_name=property_aliases_collection,
        embedding_field_name="alias_text_embedding",
        index_name=property_aliases_index,
        dimensions=embedding_dimensions,
    )
    logger.info("Search index created successfully")
    logger.info("All indexes created successfully")

    return db


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create collections and indexes for the dynamic triplets database"
    )
    parser.add_argument(
        "--mongo_uri",
        type=str,
        default="mongodb://localhost:27018/?directConnection=true",
    )
    parser.add_argument("--db_name", type=str, default="triplets_db")
    parser.add_argument(
        "--entity_aliases_collection",
        type=str,
        default="entity_aliases",
        help="Collection name for entity aliases",
    )
    parser.add_argument(
        "--property_aliases_collection",
        type=str,
        default="property_aliases",
        help="Collection name for property aliases",
    )
    parser.add_argument(
        "--triplets_collection",
        type=str,
        default="triplets",
        help="Collection name for triplets",
    )
    parser.add_argument(
        "--initial_triplets_collection",
        type=str,
        default="initial_triplets",
        help="Collection name for initial triplets",
    )
    parser.add_argument(
        "--filtered_triplets_collection",
        type=str,
        default="filtered_triplets",
        help="Collection name for filtered triplets",
    )
    parser.add_argument(
        "--entity_aliases_index",
        type=str,
        default="entity_aliases",
        help="Index name for entities",
    )
    parser.add_argument(
        "--property_aliases_index",
        type=str,
        default="property_aliases",
        help="Index name for property aliases",
    )
    parser.add_argument(
        "--embedding_dimensions",
        type=int,
        default=768,
        help="Dimensions for embeddings",
    )
    parser.add_argument(
        "--drop_collections", type=bool, default=False, help="Drop existing collections"
    )

    args = parser.parse_args()
    create_triplets_database(
        mongo_uri=args.mongo_uri,
        db_name=args.db_name,
        entity_aliases_collection=args.entity_aliases_collection,
        property_aliases_collection=args.property_aliases_collection,
        triplets_collection=args.triplets_collection,
        initial_triplets_collection=args.initial_triplets_collection,
        filtered_triplets_collection=args.filtered_triplets_collection,
        entity_aliases_index=args.entity_aliases_index,
        property_aliases_index=args.property_aliases_index,
        embedding_dimensions=args.embedding_dimensions,
        drop_collections=args.drop_collections,
    )
