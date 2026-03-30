import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC


@OPERATOR_REGISTRY.register()
class MedKGGetTestMeasurementOntology(OperatorABC):

    def __init__(self):
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "MedKGGetTestMeasurementOntology 用于加载检查与检验本体。",
                "包含实体类型、关系类型、属性类型。",
                "输出: entity_type, relation_type, attribute_type"
            )
        return (
            "MedKGGetTestMeasurementOntology loads the test and measurement ontology for MedKG.",
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
                "Symptom",
                "Finding",
                "Laboratory or Test Result",
                "Clinical Attribute",
                "Organism Attribute"
            ],
            "Substance_and_Drug": [
                "Compound",
                "Body Substance",
                "Chemical",
                "Clinical Drug",
                "Nucleic Acid, Nucleoside, or Nucleotide",
                "Amino Acid, Peptide, or Protein"
            ],
            "Medical_Activity": [
                "Laboratory Test",
                "Laboratory Procedure",
                "Diagnostic Procedure",
                "Molecular Biology Research Technique",
                "Procedure"
            ],
            "Physical_Object": [
                "Medical Device"
            ],
            "Anatomy_and_Organism": [
                "Anatomy",
                "Anatomical Structure",
                "Body System",
                "Body Part, Organ, or Organ Component",
                "Tissue",
                "Cell",
                "Body Location or Region",
                "Body Space or Junction"
            ],
            "Molecular_Biology": [
                "Gene",
                "Gene or Genome",
                "Biological Process",
                "Cellular Component",
                "Cell Component",
                "Molecular Function",
                "Molecular Sequence",
                "Nucleotide Sequence",
                "Amino Acid Sequence"
            ]
        }

        return entity_types

    # =========================
    # Relation Ontology
    # =========================

    def load_relation_types(self):

        relation_types = relation_types = {
            "Anatomy-Laboratory Test Relation": [
                "location_of"
            ],
            "Anatomy-Procedure Relation": [
                "location_of"
            ],
            "Cellular Component-Laboratory Test Relation": [
                "location_of"
            ],
            "Cellular Component-Procedure Relation": [
                "location_of"
            ],
            "Laboratory Test-Compound Relation": [
                "analyzes", "measures", "assesses_effect_of", "uses"
            ],
            "Laboratory Test-Disease Relation": [
                "diagnoses"
            ],
            "Laboratory Test-Clinical Attribute Relation": [
                "measures"
            ],
            "Laboratory Test-Laboratory Test Relation": [
                "method_of"
            ],
            "Laboratory Test-Procedure Relation": [
                "precedes", "method_of"
            ],
            "Laboratory Test Result-Laboratory Test Result Relation": [
                "conceptual_part_of"
            ],
            "Laboratory Test Result-Disease Relation": [
                "indicates"
            ],
            "Laboratory Test Result-Biological Process Relation": [
                "indicates", "measurement_of"
            ],
            "Laboratory Test Result-Compound Relation": [
                "measurement_of"
            ],
            "Laboratory Test Result-Laboratory Test Relation": [
                "result_of"
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
            file_path="./.cache/medical/test_measurement_ontology.json",
            use_current_step=False
        )

        self.logger.info(f"Ontology saved to {output_file}")

        return ["entity_type", "relation_type"]
