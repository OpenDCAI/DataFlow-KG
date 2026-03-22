# -*- coding: utf-8 -*-
"""
====================================
DataFlow-KG: GeoKGGetBasicOntology
====================================

Author: Zhengpin Li
Affiliation: Peking University
Email: zpli@pku.edu.cn
Created: 2026-03-05
"""

import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC


@OPERATOR_REGISTRY.register()
class GeoKGGetBasicOntology(OperatorABC):

    def __init__(self):
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "GeoKGGetBasicOntology 用于加载地理知识图谱基础本体。",
                "包含实体类型、关系类型、属性类型及时空类型。",
                "输出: entity_type, relation_type, attribute_type, temporal_type"
            )
        return (
            "GeoKGGetBasicOntology loads the basic ontology for GeoKG.",
            "Includes entity types, relation types, attribute types, and temporal types.",
            "Output: entity_type, relation_type, attribute_type, temporal_type"
        )

    # =========================
    # Entity Ontology
    # =========================

    def load_entity_types(self):

        entity_types = {

            "NaturalFeature": [
                "Mountain","Volcano","Plateau","Valley",
                "River","Lake","Ocean","Glacier","Desert","Forest"
            ],

            "AdministrativeRegion": [
                "Country","State","Province","Prefecture",
                "City","County","District","Town","Village"
            ],

            "Infrastructure": [
                "Road","Highway","Railway","Bridge",
                "Tunnel","Airport","Port","Dam","Canal"
            ]
        }

        return entity_types

    # =========================
    # Relation Ontology
    # =========================

    def load_relation_types(self):

        relation_types = {

            "SpatialRelation": [
                "located_in","part_of","adjacent_to",
                "near","contains","within"
            ],

            "TopologicalRelation": [
                "intersects","touches","crosses","disjoint"
            ],

            "HydrologicalRelation": [
                "flows_through","flows_into","originates_from","tributary_of"
            ],

            "AdministrativeRelation": [
                "capital_of","governs","administers","belongs_to_region"
            ],

            "InfrastructureRelation": [
                "connected_by","served_by","accessible_via"
            ],

            "TemporalRelation": [
                "existed_during","built_in","founded_in","abolished_in"
            ]
        }

        return relation_types

    # =========================
    # Attribute Ontology
    # =========================

    def load_attribute_types(self):

        attribute_types = {

            "SpatialAttribute": [
                "latitude","longitude","elevation",
                "area","length","width","depth"
            ],

            "AdministrativeAttribute": [
                "population","population_density",
                "postal_code","administrative_code"
            ],

            "EnvironmentalAttribute": [
                "climate_type","average_temperature",
                "annual_rainfall","vegetation_type"
            ],

            "EconomicAttribute": [
                "GDP","GDP_per_capita","major_industry"
            ],

            "TemporalAttribute": [
                "established_date","construction_date","dissolution_date",
                "historical_period","observation_time"
            ]
        }

        return attribute_types


    # =========================
    # Run
    # =========================

    def run(self, 
        storage: DataFlowStorage = None
        ):

        self.logger.info("Loading GeoKG ontology")

        entity_types = self.load_entity_types()
        relation_types = self.load_relation_types()
        attribute_types = self.load_attribute_types()

        dataframe = pd.DataFrame({
            "entity_type": [entity_types],
            "relation_type": [relation_types],
            "attribute_type": [attribute_types],
        })

        output_file = storage.write(dataframe, file_path="./.cache/api/ontology.json", use_current_step=False)

        self.logger.info(f"Ontology saved to {output_file}")

        return ["entity_type","relation_type","attribute_type","temporal_type"]
