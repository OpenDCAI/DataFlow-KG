import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC


@OPERATOR_REGISTRY.register()
class MedKGGetAnatomyBiologyOntology(OperatorABC):

    def __init__(self):
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "MedKGGetAnatomyBiologyOntology 用于加载解剖与生物本体。",
                "包含实体类型、关系类型、属性类型。",
                "输出: entity_type, relation_type, attribute_type"
            )
        return (
            "MedKGGetAnatomyBiologyOntology loads the anatomy and biology ontology for MedKG.",
            "Includes entity types, relation types, attribute types.",
            "Output: entity_type, relation_type, attribute_type"
        )

    # =========================
    # Entity Ontology
    # =========================

    def load_entity_types(self):

        entity_types = {
            "Clinical_Concept": [],
            "Substance_and_Drug": [
                "Compound"
            ],
            "Medical_Activity": [],
            "Physical_Object": [],
            "Anatomy_and_Organism": [
                "Anatomy",
                "Organism",
                "Human",
                "Anatomical Structure",
                "Embryonic Structure",
                "Fully Formed Anatomical Structure",
                "Body System",
                "Body Part, Organ, or Organ Component",
                "Tissue",
                "Cell",
                "Body Location or Region",
                "Body Space or Junction"
            ],
            "Molecular_Biology": [
                "Gene",
                "Pathway",
                "Biological Process",
                "Cellular Component",
                "Molecular Function",
                "Gene or Genome",
                "Cell Component",
                "Biologic Function",
                "Physiologic Function",
                "Organism Function",
                "Mental Process",
                "Organ or Tissue Function",
                "Cell Function",
                "Genetic Function",
                "Molecular Sequence",
                "Nucleotide Sequence",
                "Amino Acid Sequence",
                "Carbohydrate Sequence"
            ]
        }

        return entity_types

    # =========================
    # Relation Ontology
    # =========================

    def load_relation_types(self):

        relation_types = relation_types = {
            "Anatomy-Gene Relation": [
                "downregulates", "expresses", "upregulates"
            ],
            "Gene-Biological Process Relation": [
                "participates", "affects"
            ],
            "Gene-Cellular Component Relation": [
                "participates", "part_of"
            ],
            "Gene-Gene Relation": [
                "covaries", "interacts", "regulates", "property_of", "consists_of", "part_of"
            ],
            "Gene-Molecular Function Relation": [
                "participates", "carries_out"
            ],
            "Gene-Pathway Relation": [
                "participates"
            ],
            "Gene-Compound Relation": [
                "property_of"
            ],
            "Biological Process-Organism Relation": [
                "affects", "process_of"
            ],
            "Biological Process-Compound Relation": [
                "produces"
            ],
            "Anatomy-Anatomy Relation": [
                "adjacent_to", "conceptual_part_of", "connected_to", "location_of",
                "traverses", "branch_of", "consists_of", "developmental_form_of",
                "part_of", "surrounds", "tributary_of", "interconnects", "contains"
            ],
            "Anatomy-Biological Process Relation": [
                "location_of"
            ],
            "Anatomy-Compound Relation": [
                "connected_to", "consists_of", "contains", "location_of", "produces", "surrounds"
            ],
            "Anatomy-Cellular Component Relation": [
                "location_of", "surrounds"
            ],
            "Cellular Component-Anatomy Relation": [
                "adjacent_to", "location_of", "conceptual_part_of", "part_of"
            ],
            "Cellular Component-Cellular Component Relation": [
                "adjacent_to", "part_of", "surrounds"
            ],
            "Cellular Component-Biological Process Relation": [
                "affects"
            ],
            "Gene-Anatomy Relation": [
                "part_of"
            ],
            "Biological Process-Cellular Component Relation": [
                "produces"
            ],
            "Biological Process-Biological Process Relation": [
                "degree_of", "occurs_in", "co-occurs_with", "precedes"
            ],
            "Biological Process-Clinical Attribute Relation": [
                "affects"
            ],
            "Anatomy-Organism Relation": [
                "part_of"
            ],
            "Organism-Organism Relation": [
                "interacts_with"
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
            file_path="./.cache/medical/anatomy_biology_ontology.json",
            use_current_step=False
        )

        self.logger.info(f"Ontology saved to {output_file}")

        return ["entity_type", "relation_type"]
