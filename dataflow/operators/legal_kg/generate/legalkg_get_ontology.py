# -*- coding: utf-8 -*-
"""
====================================
DataFlow-KG: LegalKGGetBasicOntology
====================================

Author: Zhengpin Li
Description: General Legal Knowledge Graph Ontology
Supports criminal, civil, administrative cases
"""

import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC


@OPERATOR_REGISTRY.register()
class LegalKGGetBasicOntology(OperatorABC):

    def __init__(self):
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "zh") -> tuple:
        return (
            "LegalKGGetBasicOntology：通用法律知识图谱本体",
            "覆盖刑事/民事/行政案件，支持事件建模",
            "输出: entity_type, relation_type, attribute_type"
        )

    # =========================
    # Entity Ontology
    # =========================
    def load_entity_types(self):

        entity_types = {

            # 主体
            "LegalActor": [
                "NaturalPerson",
                "LegalPerson",
                "Organization"
            ],

            "LitigationRole": [
                "Plaintiff",
                "Defendant",
                "Victim",
                "Appellant",
                "Appellee",
                "Prosecutor",
                "Judge",
                "Lawyer",
                "ThirdParty"
            ],

            # 机构
            "LegalInstitution": [
                "Court",
                "Procuratorate",
                "PublicSecurity",
                "AdministrativeAgency",
                "DetentionCenter",
                "Prison",
                "LawFirm"
            ],

            # 案件
            "Case": [
                "CriminalCase",
                "CivilCase",
                "AdministrativeCase",
                "EnforcementCase"
            ],

            # 法律概念
            "LegalConcept": [
                "Crime",
                "CauseOfAction",
                "LegalRight",
                "LegalObligation",
                "LawArticle",
                "Evidence",
                "Charge",
                "Judgment",
                "Ruling"
            ],

            # 事件
            "LegalEvent": [
                "IllegalAct",
                "CivilAct",
                "ProceduralAct",
                "ContractAct",
                "TortAct"
            ],

            # 客体
            "Object": [
                "Property",
                "Money",
                "Goods",
                "Service",
                "IntellectualProperty"
            ],

            # 时空
            "Time": ["DateTime"],
            "Location": ["Place", "City", "Region"]
        }

        return entity_types

    # =========================
    # Relation Ontology
    # =========================
    def load_relation_types(self):

        relation_types = {

            "CaseRelation": [
                "has_party",
                "handled_by",
                "filed_by",
                "appealed_to",
                "belongs_to_case_type"
            ],

            "RoleRelation": [
                "plays_role",
                "represents",
                "defends",
                "prosecutes"
            ],

            "ActionRelation": [
                "commits",
                "against",
                "affects",
                "results_in"
            ],

            "LegalDetermination": [
                "constitutes",
                "violates",
                "based_on",
                "supported_by"
            ],

            "JudicialRelation": [
                "judged_by",
                "sentenced_to",
                "ordered_to_pay",
                "recognized",
                "dismissed",
                "revoked"
            ],

            "ProceduralRelation": [
                "filed",
                "accepted",
                "heard",
                "detained",
                "arrested",
                "released_on_bail",
                "appealed",
                "executed"
            ],

            "RightsObligations": [
                "has_right",
                "has_obligation",
                "breaches",
                "fulfills"
            ],

            "TemporalRelation": [
                "occurs_at",
                "starts_at",
                "ends_at"
            ],

            "SpatialRelation": [
                "occurs_in",
                "located_in"
            ]
        }

        return relation_types

    # =========================
    # Attribute Ontology
    # =========================
    def load_attribute_types(self):

        attribute_types = {

            "ActorAttribute": [
                "name",
                "gender",
                "birth_date",
                "id_number",
                "address",
                "occupation"
            ],

            "CaseAttribute": [
                "case_id",
                "case_type",
                "trial_procedure",
                "court_level"
            ],

            "EventAttribute": [
                "amount",
                "value",
                "means",
                "intent",
                "consequence"
            ],

            "EvidenceAttribute": [
                "evidence_type",
                "source",
                "credibility"
            ],

            "JudgmentAttribute": [
                "sentence_length",
                "fine_amount",
                "compensation_amount",
                "probation",
                "judgment_result",
                "effective_date"
            ],

            "ProcedureAttribute": [
                "stage",
                "status"
            ],

            "TemporalAttribute": [
                "timestamp",
                "duration"
            ]
        }

        return attribute_types

    # =========================
    # Run
    # =========================
    def run(self, storage: DataFlowStorage = None):

        self.logger.info("Loading LegalKG ontology")

        entity_types = self.load_entity_types()
        relation_types = self.load_relation_types()
        attribute_types = self.load_attribute_types()

        dataframe = pd.DataFrame({
            "entity_type": [entity_types],
            "relation_type": [relation_types],
            "attribute_type": [attribute_types],
        })

        output_file = storage.write(
            dataframe,
            file_path="./.cache/api/legal_ontology.json",
            use_current_step=False
        )

        self.logger.info(f"Ontology saved to {output_file}")

        return ["entity_type", "relation_type", "attribute_type"]