from dataflow.serving.api_llm_serving_request import APILLMServing_request
from dataflow.utils.storage import FileStorage
from dataflow.operators.domain_kg.utils.legalkg_get_ontology import (
    LegalKGGetBasicOntology,
)
from dataflow.operators.domain_kg.legal_kg.generate.legalkg_triple_extractor import (
    LegalKGTupleExtraction,
)

from dataflow.operators.domain_kg.legal_kg.eval.legalkg_case_similarity_eval import (
    LegalKGCaseSummarySimilarity,
)
from dataflow.operators.domain_kg.legal_kg.filter.legalkg_case_similarity_filtering import (
    LegalKGCaseSimilarityFilter,
)
from dataflow.operators.domain_kg.legal_kg.generate.legalkg_case_judgement_generator import (
    LegalKGJudgementPrediction,
)


class LegalKGPipeline:
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/LegalKGPipeline/input.json",
            cache_path="./cache_legalkg",
            file_name_prefix="legal_kg_pipeline",
            cache_type="jsonl",
        )
        self.ontology_storage = FileStorage(
            first_entry_file_name="",
            cache_path="./cache_legalkg_ontology",
            file_name_prefix="legal_kg_ontology",
            cache_type="json",
        )

        self.llm_serving = APILLMServing_request(
            api_url="http://123.129.219.111:3000/v1/chat/completions",
            model_name="gpt-4o",
            max_workers=20,
        )

        self.ontology_loader_step1 = LegalKGGetBasicOntology(lang="en")
        self.triple_extractor_step2 = LegalKGTupleExtraction(
            llm_serving=self.llm_serving,
            triple_type="relation",
            lang="en",
        )
        self.case_similarity_eval_step3 = LegalKGCaseSummarySimilarity(
            llm_serving=self.llm_serving,
            lang="en",
        )
        self.case_similarity_filter_step4 = LegalKGCaseSimilarityFilter()
        self.judgement_prediction_step5 = LegalKGJudgementPrediction(
            llm_serving=self.llm_serving,
            lang="en",
        )

    def forward(self):
        self.ontology_loader_step1.run(
            storage=self.ontology_storage.step(),
        )

        self.triple_extractor_step2.run(
            storage=self.storage.step(),
            input_key="raw_chunk",
            input_key_meta="legal_ontology",
            output_key="triple",
            output_key_meta1="entity_class",
            output_key_meta2="case_summary",
        )

        self.case_similarity_eval_step3.run(
            storage=self.storage.step(),
            input_key="case_summary",
            input_key_meta=["theft case", "theft case"],
            output_key="similarity_score",
        )

        self.case_similarity_filter_step4.run(
            storage=self.storage.step(),
            input_key="triple",
            output_key="similarity_score",
            min_score=0.6,
            max_score=1.0,
        )

        self.judgement_prediction_step5.run(
            storage=self.storage.step(),
            input_key="triple",
            input_key_meta=["Zhang San stole an iPhone worth RMB 6000"],
            output_key_judgement="judgement",
            output_key_reason="reason",
        )


if __name__ == "__main__":
    pipeline = LegalKGPipeline()
    pipeline.forward()
