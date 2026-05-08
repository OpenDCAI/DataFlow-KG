from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage

from dataflow.operators.general_kg import KGQANaturalEvaluator
from dataflow.operators.hyper_relation_kg import HRKGRelationTripleAttributeFilter
from dataflow.operators.hyper_relation_kg import (
    HRKGTripleExtraction,
)
from dataflow.operators.hyper_relation_kg import (
    HRKGRelationTripleSubgraphQAGeneration,
)


class HyperKGQA_APIPipeline:
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/HyperRelationKGPipeline/input.json",
            cache_path="./hyper_kg",
            file_name_prefix="hyper_kg_qa_pipeline",
            cache_type="json",
        )

        self.llm_serving = APILLMServing_request(
            api_url="http://123.129.219.111:3000/v1/chat/completions",
            model_name="gpt-4o",
            max_workers=20,
        )

        self.hyper_triple_extraction_step1 = HRKGTripleExtraction(
            llm_serving=self.llm_serving,
            lang="en",
        )

        self.subgraph_filter_step2 = HRKGRelationTripleAttributeFilter(
            lang="en",
        )

        self.subgraph_qa_generation_step3 = HRKGRelationTripleSubgraphQAGeneration(
            llm_serving=self.llm_serving,
            lang="en",
            qa_type="num",
            num_q=5,
        )

        self.qa_natural_eval_step4 = KGQANaturalEvaluator(
            llm_serving=self.llm_serving,
            lang="en",
        )

    def forward(self):
        self.hyper_triple_extraction_step1.run(
            storage=self.storage.step(),
            input_key="text",
            output_key="tuple",
        )

        self.subgraph_filter_step2.run(
            storage=self.storage.step(),
            input_key="tuple",
            output_key="subgraph",
            attr_tag="<Time>",
        )

        self.subgraph_qa_generation_step3.run(
            storage=self.storage.step(),
            input_key="subgraph",
            output_key="QA_pairs",
        )

        self.qa_natural_eval_step4.run(
            storage=self.storage.step(),
            input_key="QA_pairs",
            output_key="naturalness_scores",
        )


if __name__ == "__main__":
    model = HyperKGQA_APIPipeline()
    model.forward()
