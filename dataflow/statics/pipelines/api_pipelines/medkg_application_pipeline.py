from dataflow.serving.api_llm_serving_request import APILLMServing_request
from dataflow.utils.storage import FileStorage
from dataflow.operators.domain_kg.utils.medkg_get_drug_therapy_ontology import (
    MedKGGetDrugTherapyOntology,
)
from dataflow.operators.domain_kg.medical_kg.generate.medkg_triple_extractor import (
    MedKGTripleExtraction,
)
from dataflow.operators.domain_kg.medical_kg.generate.medkg_triple_drug_action_mechanism_discovery import (
    MedKGTripleDrugActionMechanismDiscovery,
)


class MedicalKGPipeline:
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/MedicalKGPipeline/input.json",
            cache_path="./cache_medkg",
            file_name_prefix="medical_kg_pipeline",
            cache_type="jsonl",
        )
        self.ontology_storage = FileStorage(
            first_entry_file_name="",
            cache_path="./cache_medkg_ontology",
            file_name_prefix="medical_kg_ontology",
            cache_type="json",
        )

        self.llm_serving = APILLMServing_request(
            api_url="http://123.129.219.111:3000/v1/chat/completions",
            model_name="gpt-4o",
            max_workers=20,
        )

        self.ontology_loader_step1 = MedKGGetDrugTherapyOntology()
        self.triple_extractor_step2 = MedKGTripleExtraction(
            llm_serving=self.llm_serving,
            lang="en",
        )
        self.mechanism_discovery_step3 = MedKGTripleDrugActionMechanismDiscovery(
            llm_serving=self.llm_serving,
            lang="en",
            max_hop=3,
            max_candidate_paths=20,
        )

    def forward(self):
        self.ontology_loader_step1.run(
            storage=self.ontology_storage.step(),
        )

        self.triple_extractor_step2.run(
            storage=self.storage.step(),
            input_key="raw_chunk",
            input_key_meta="drug_therapy_ontology",
            output_key="triple",
            output_key_meta="entity_class",
        )


        self.mechanism_discovery_step3.run(
            storage=self.storage.step(),
            input_key_query="query",
            input_key_triple="triple",
            output_key_path="mechanism_path",
            output_key_answer="mechanism_answer",
        )


if __name__ == "__main__":
    pipeline = MedicalKGPipeline()
    pipeline.forward()
