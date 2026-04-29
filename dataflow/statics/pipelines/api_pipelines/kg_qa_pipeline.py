from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage

from dataflow.operators.general_kg.eval.kg_qa_natural_eval import KGQANaturalEvaluator
from dataflow.operators.general_kg.eval.kg_subgraph_scale_eval import KGSubgraphScaleEvaluator
from dataflow.operators.general_kg.filter.kg_rel_tuple_subgraph_sampling import (
    KGEntityBasedSubgraphSampling,
)
from dataflow.operators.general_kg.filter.kg_subgraph_scale_filtering import (
    KGSubgraphScaleFilter,
)
from dataflow.operators.general_kg.filter.kg_tuple_remove_repeated import (
    KGTupleRemoveRepeated,
)
from dataflow.operators.general_kg.generate.kg_entity_extractor import KGEntityExtraction
from dataflow.operators.general_kg.generate.kg_rel_triple_subgraph_qa_generator import (
    KGRelationTripleSubgraphQAGeneration,
)
from dataflow.operators.general_kg.generate.kg_triple_extractor import KGTripleExtraction


class KGQA_APIPipeline:
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/KG2QAPipeline/input.json",
            cache_path="./qa_generation",
            file_name_prefix="kg_qa_pipeline",
            cache_type="json",
        )

        self.llm_serving = APILLMServing_request(
            api_url="http://123.129.219.111:3000/v1/chat/completions",
            model_name="gpt-4o",
            max_workers=10,
        )

        self.entity_extraction_step1 = KGEntityExtraction(
            llm_serving=self.llm_serving,
            lang="en",
        )

        self.triple_extraction_step2 = KGTripleExtraction(
            llm_serving=self.llm_serving,
            triple_type="relation",
            lang="en",
        )

        self.triple_dedup_step3 = KGTupleRemoveRepeated()

        self.subgraph_sampling_step4 = KGEntityBasedSubgraphSampling(
            llm_serving=self.llm_serving,
            lang="en",
        )

        self.subgraph_scale_eval_step5 = KGSubgraphScaleEvaluator()

        self.subgraph_scale_filter_step6 = KGSubgraphScaleFilter()

        self.subgraph_qa_generation_step7 = KGRelationTripleSubgraphQAGeneration(
            llm_serving=self.llm_serving,
            lang="en",
            qa_type="num",
            num_q=5,
        )

        self.qa_natural_eval_step8 = KGQANaturalEvaluator(
            llm_serving=self.llm_serving,
            lang="en",
        )

    def forward(self):
        self.entity_extraction_step1.run(
            storage=self.storage.step(),
            input_key="text",
            output_key="entity",
        )

        self.triple_extraction_step2.run(
            storage=self.storage.step(),
            input_key="text",
            input_key_meta="entity",
            output_key="triple",
        )

        self.triple_dedup_step3.run(
            storage=self.storage.step(),
            input_key="triple",
            output_key="triple",
        )

        self.subgraph_sampling_step4.run(
            storage=self.storage.step(),
            input_key="triple",
            output_key="subgraph",
            sampling_type="hop",
            hop=2,
            M=5,
        )

        self.subgraph_scale_eval_step5.run(
            storage=self.storage.step(),
            input_key="subgraph",
            output_key1="num_nodes",
            output_key2="num_edges",
            output_key3="density",
        )

        self.subgraph_scale_filter_step6.run(
            storage=self.storage.step(),
            input_key="subgraph",
            output_key="density",
            min_score=0.1,
            max_score=1.0,
        )

        self.subgraph_qa_generation_step7.run(
            storage=self.storage.step(),
            input_key="subgraph",
            output_key="QA_pairs",
        )

        self.qa_natural_eval_step8.run(
            storage=self.storage.step(),
            input_key="QA_pairs",
            output_key="naturalness_scores",
        )


if __name__ == "__main__":
    model = KGQA_APIPipeline()
    model.forward()
