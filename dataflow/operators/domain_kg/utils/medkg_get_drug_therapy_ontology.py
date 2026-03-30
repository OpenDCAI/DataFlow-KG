import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC


@OPERATOR_REGISTRY.register()
class MedKGGetDrugTherapyOntology(OperatorABC):

    def __init__(self):
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "MedKGGetDrugTherapyOntology 用于加载药物与治疗本体。",
                "包含实体类型、关系类型、属性类型。",
                "输出: entity_type, relation_type, attribute_type"
            )
        return (
            "MedKGGetDrugTherapyOntology loads the drug and therapy ontology for MedKG.",
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
                "Disease or Syndrome"
            ],
            "Substance_and_Drug": [
                "Compound",
                "Pharmacologic Class",
                "Substance",
                "Body Substance",
                "Chemical",
                "Chemical Viewed Structurally",
                "Organic Chemical",
                "Chemical Viewed Functionally",
                "Pharmacologic Substance",
                "Biomedical or Dental Material",
                "Biologically Active Substance",
                "Hormone",
                "Enzyme",
                "Vitamin",
                "Immunologic Factor",
                "Indicator, Reagent, or Diagnostic Aid",
                "Hazardous or Poisonous Substance",
                "Food",
                "Receptor",
                "Antibiotic",
                "Element, Ion, or Isotope",
                "Inorganic Chemical",
                "Clinical Drug",
                "Nucleic Acid, Nucleoside, or Nucleotide",
                "Amino Acid, Peptide, or Protein"
            ],
            "Medical_Activity": [
                "Procedure",
                "Therapeutic or Preventive Procedure"
            ],
            "Physical_Object": [
                "Medical Device",
                "Drug Delivery Device"
            ],
            "Anatomy_and_Organism": [
                "Anatomy"
            ],
            "Molecular_Biology": [
                "Gene",
                "Gene or Genome",
                "Biological Process",
                "Molecular Function"
            ]
        }

        return entity_types

    # =========================
    # Relation Ontology
    # =========================

    def load_relation_types(self):

        relation_types = relation_types = {
            "Compound-Compound Relation": [
                "resembles", "consists_of", "part_of", "interacts_with", "ingredient_of"
            ],
            "Compound-Disease Relation": [
                "palliates", "treats", "complicates", "indicates", "diagnoses", "prevents", "causes"
            ],
            "Compound-Gene Relation": [
                "binds", "downregulates", "upregulates"
            ],
            "Compound-Side Effect Relation": [
                "causes"
            ],
            "Pharmacologic Class-Compound Relation": [
                "includes"
            ],
            "Gene-Compound Relation": [
                "property_of"
            ],
            "Biological Process-Compound Relation": [
                "produces"
            ],
            "Compound-Biological Process Relation": [
                "complicates", "disrupts", "affects"
            ],
            "Compound-Anatomy Relation": [
                "disrupts", "conceptual_part_of", "derivative_of", "part_of", "surrounds"
            ],
            "Anatomy-Compound Relation": [
                "connected_to", "consists_of", "contains", "location_of", "produces", "surrounds"
            ],
            "Medical Device-Compound Relation": [
                "contains"
            ],
            "Medical Device-Disease Relation": [
                "prevents", "treats"
            ],
            "Medical Device-Symptom Relation": [
                "treats"
            ],
            "Compound-Symptom Relation": [
                "treats"
            ],
            "Procedure-Biological Process Relation": [
                "complicates"
            ],
            "Procedure-Procedure Relation": [
                "method_of"
            ],
            "Procedure-Disease Relation": [
                "prevents", "treats"
            ],
            "Procedure-Symptom Relation": [
                "treats"
            ],
            "Procedure-Compound Relation": [
                "uses"
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
            file_path="./.cache/medical/drug_therapy_ontology.json",
            use_current_step=False
        )

        self.logger.info(f"Ontology saved to {output_file}")

        return ["entity_type", "relation_type"]
