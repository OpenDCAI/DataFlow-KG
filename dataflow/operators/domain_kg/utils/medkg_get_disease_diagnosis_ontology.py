import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC


@OPERATOR_REGISTRY.register()
class MedKGGetDiseaseDiagnosisOntology(OperatorABC):

    def __init__(self):
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "MedKGGetDiseaseDiagnosisOntology 用于加载疾病与诊断本体。",
                "包含实体类型、关系类型、属性类型。",
                "输出: entity_type, relation_type, attribute_type"
            )
        return (
            "MedKGGetDiseaseDiagnosisOntology loads the disease and diagnosis ontology for MedKG.",
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
                "Side Effect",
                "Disease or Syndrome",
                "Sign or Symptom",
                "Finding",
                "Laboratory or Test Result",
                "Clinical Attribute",
                "Organism Attribute",
                "Anatomical Abnormality",
                "Congenital Abnormality",
                "Acquired Abnormality",
                "Injury or Poisoning",
                "Pathologic Function",
                "Mental or Behavioral Dysfunction",
                "Cell or Molecular Dysfunction",
                "Neoplastic Process",
                "Experimental Model of Disease"
            ],
            "Substance_and_Drug": [],
            "Medical_Activity": [
                "Laboratory Test",
                "Laboratory Procedure",
                "Diagnostic Procedure"
            ],
            "Physical_Object": [],
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
                "Biologic Function",
                "Physiologic Function"
            ]
        }

        return entity_types

    # =========================
    # Relation Ontology
    # =========================

    def load_relation_types(self):

        relation_types = relation_types = {
            "Disease-Anatomy Relation": [
                "localizes", "produces", "disrupts"
            ],
            "Disease-Disease Relation": [
                "resembles", "co-occurs_with", "complicates", "result_of",
                "location_of", "conceptually_related_to", "associated_with",
                "degree_of", "manifestation_of", "precedes", "occurs_in"
            ],
            "Disease-Gene Relation": [
                "associates", "downregulates", "upregulates"
            ],
            "Disease-Symptom Relation": [
                "presents"
            ],
            "Disease-Organism Relation": [
                "affects"
            ],
            "Disease-Biological Process Relation": [
                "affects", "manifestation_of", "disrupts"
            ],
            "Disease-Clinical Attribute Relation": [
                "associated_with"
            ],
            "Anatomy-Disease Relation": [
                "location_of"
            ],
            "Laboratory Test-Disease Relation": [
                "diagnoses"
            ],
            "Laboratory Test-Clinical Attribute Relation": [
                "measures"
            ],
            "Symptom-Disease Relation": [
                "associated_with", "manifestation_of", "diagnoses"
            ],
            "Symptom-Symptom Relation": [
                "co-occurs_with", "degree_of"
            ],
            "Symptom-Biological Process Relation": [
                "evaluation_of", "manifestation_of"
            ],
            "Symptom-Clinical Attribute Relation": [
                "evaluation_of"
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
            "Laboratory Test Result-Laboratory Test Relation": [
                "result_of"
            ],
            "Clinical Attribute-Clinical Attribute Relation": [
                "associated_with", "degree_of"
            ],
            "Clinical Attribute-Biological Process Relation": [
                "manifestation_of", "measurement_of"
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
            file_path="./.cache/medical/disease_diagnosis_ontology.json",
            use_current_step=False
        )

        self.logger.info(f"Ontology saved to {output_file}")

        return ["entity_type", "relation_type"]
