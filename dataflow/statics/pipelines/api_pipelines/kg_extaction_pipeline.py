from dataflow.serving.api_llm_serving_request import APILLMServing_request
from dataflow.utils.storage import FileStorage
from dataflow.operators.general_kg.generate.kg_entity_extractor import (
    KGEntityExtraction,
)
from dataflow.operators.general_kg.generate.kg_triple_extractor import (
    KGTripleExtraction,
)
from dataflow.operators.general_kg.generate.kg_rel_triple_inference import (
    KGRelationTripleInference,
)
from dataflow.operators.general_kg.filter.kg_tuple_remove_repeated import (
    KGTupleRemoveRepeated,
)
from dataflow.operators.general_kg.filter.kg_tuple_validation import (
    KGTupleValidity,
)


class KGExtractionPipeline:
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/kg_extraction_pipeline_input.json",
            cache_path="./cache_kg_extraction",
            file_name_prefix="kg_extraction_pipeline",
            cache_type="jsonl",
        )

        self.llm_serving = APILLMServing_request(
            api_url="http://123.129.219.111:3000/v1/chat/completions",
            model_name="gpt-4o-mini",
            max_workers=20,
        )

        self.entity_extractor_step1 = KGEntityExtraction(
            llm_serving=self.llm_serving,
            lang="en",
        )
        self.triple_extractor_step2 = KGTripleExtraction(
            llm_serving=self.llm_serving,
            triple_type="relation",
            lang="en",
        )
        self.triple_inference_step3 = KGRelationTripleInference(
            llm_serving=self.llm_serving,
            lang="en",
            merge_to_input=True,
        )
        self.tuple_dedup_step4 = KGTupleRemoveRepeated(
            lang="en",
        )
        self.tuple_validation_step5 = KGTupleValidity(
            llm_serving=self.llm_serving,
            lang="en",
            triple_type="relation",
        )

    def forward(self):
        self.entity_extractor_step1.run(
            storage=self.storage.step(),
            input_key="raw_chunk",
            output_key="entity",
        )

        self.triple_extractor_step2.run(
            storage=self.storage.step(),
            input_key="raw_chunk",
            input_key_meta="entity",
            output_key="triple",
        )

        self.triple_inference_step3.run(
            storage=self.storage.step(),
            input_key="triple",
            output_key="inferred_triple",
        )

        self.tuple_dedup_step4.run(
            storage=self.storage.step(),
            input_key="triple",
            output_key="triple",
        )

        self.tuple_validation_step5.run(
            storage=self.storage.step(),
            input_key="triple",
            output_key="valid_triple",
        )


if __name__ == "__main__":
    model = KGExtractionPipeline()
    model.forward()
