from dataflow.serving.api_llm_serving_request import APILLMServing_request
from dataflow.utils.storage import FileStorage
from dataflow.operators.domain_kg.utils import SchoKGGetOntology
from dataflow.operators.domain_kg.scholar_kg import SchoKGTripleExtraction
from dataflow.operators.domain_kg.scholar_kg import SchoKGQueryReasoningOperator


class ScholarKGPipeline:
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/ScholarKGPipeline/input.json",
            cache_path="./schokg",
            file_name_prefix="scholar_kg_pipeline",
            cache_type="json",
        )
        self.ontology_storage = FileStorage(
            first_entry_file_name="",
            cache_path="./cache_schokg_ontology",
            file_name_prefix="scholar_kg_ontology",
            cache_type="json",
        )

        self.llm_serving = APILLMServing_request(
            api_url="http://123.129.219.111:3000/v1/chat/completions",
            model_name="gpt-4o",
            max_workers=20,
        )

        self.ontology_loader_step1 = SchoKGGetOntology()

        self.triple_extractor_step2 = SchoKGTripleExtraction(
            llm_serving=self.llm_serving,
            lang="en",
        )
        self.query_reasoning_step3 = SchoKGQueryReasoningOperator(
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
            input_key_meta="ontology",
            output_key="triple",
            output_key_meta="entity_class",
        )

        self.query_reasoning_step3.run(
            storage=self.storage.step(),
            input_key_query="query",
            input_key_triple="triple",
            output_key_path="reasoning_path",
            output_key_answer="reasoning_answer",
        )



if __name__ == "__main__":
    pipeline = ScholarKGPipeline()
    pipeline.forward()
