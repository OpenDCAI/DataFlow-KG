import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC


@OPERATOR_REGISTRY.register()
class MedKGGetMicrobePathogenOntology(OperatorABC):

    def __init__(self):
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "MedKGGetMicrobePathogenOntology 用于加载微生物与病原体本体。",
                "包含实体类型、关系类型、属性类型。",
                "输出: entity_type, relation_type, attribute_type"
            )
        return (
            "MedKGGetMicrobePathogenOntology loads the microbe and pathogen ontology for MedKG.",
            "Includes entity types, relation types, attribute types.",
            "Output: entity_type, relation_type, attribute_type"
        )

    # =========================
    # Entity Ontology
    # =========================

    def load_entity_types(self):

        entity_types = {
            "Clinical_Concept": [
                "Disease",
                "Disease or Syndrome",
                "Symptom",
                "Pathologic Function"
            ],
            "Substance_and_Drug": [
                "Compound",
                "Biologically Active Substance",
                "Immunologic Factor",
                "Pharmacologic Substance"
            ],
            "Medical_Activity": [],
            "Physical_Object": [],
            "Anatomy_and_Organism": [
                "Microbe",
                "Organism",
                "Human",
                "Fungus",
                "Virus",
                "Bacterium",
                "Anatomy",
                "Anatomical Structure",
                "Body System",
                "Body Part, Organ, or Organ Component",
                "Tissue",
                "Cell"
            ],
            "Molecular_Biology": [
                "Biological Process"
            ]
        }

        return entity_types

    # =========================
    # Relation Ontology
    # =========================

    def load_relation_types(self):

        relation_types = relation_types = {
            "Anatomy-Microbe Relation": [
                "location_of"
            ],
            "Microbe-Disease Relation": [
                "causes"
            ],
            "Microbe-Compound Relation": [
                "location_of"
            ],
            "Compound-Microbe Relation": [
                "associated_with", "indicates"
            ],
            "Biological Process-Microbe Relation": [
                "process_of"
            ],
            "Disease-Microbe Relation": [
                "process_of"
            ]
        }

        return relation_types

    # ===========================
    # Attribute Ontology
    # ===========================



    # =========================
    # Run
    # =========================

    def run(self,
        storage: DataFlowStorage = None
        ):

        self.logger.info("Loading GeoKG ontology")

        entity_types = self.load_entity_types()
        relation_types = self.load_relation_types()

        dataframe = pd.DataFrame({
            "entity_type": [entity_types],
            "relation_type": [relation_types],
        })

        output_file = storage.write(
            dataframe,
            file_path="./.cache/medical/microbe_pathogen_ontology.json",
            use_current_step=False
        )

        self.logger.info(f"Ontology saved to {output_file}")

        return ["entity_type", "relation_type"]
