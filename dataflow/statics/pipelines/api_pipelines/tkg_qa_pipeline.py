from pathlib import Path

import pandas as pd

from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage

from dataflow.operators.general_kg.eval.kg_qa_natural_eval import KGQANaturalEvaluator
from dataflow.operators.general_kg.filter.kg_rel_tuple_path_sampling import (
    KGRelationTuplePathGenerator,
)
from dataflow.operators.temporal_kg import TKGTupleExtraction
from dataflow.operators.temporal_kg import TKGTuplePathQAGeneration
from dataflow.operators.temporal_kg import TKGTupleTimeFilter

class TKGQA_APIPipeline:
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/TemporalKGPipeline/input.json",
            cache_path="./temporal_kg",
            file_name_prefix="tkg_qa_pipeline",
            cache_type="json",
        )

        self.llm_serving = APILLMServing_request(
            api_url="http://123.129.219.111:3000/v1/chat/completions",
            model_name="gpt-4o",
            max_workers=30,
        )

        self.temporal_tuple_extraction_step1 = TKGTupleExtraction(
            llm_serving=self.llm_serving,
            triple_type="relation",
            lang="en",
        )

        self.temporal_time_filter_step2 = TKGTupleTimeFilter(
            merge_to_input=True,
        )

        self.path_generation_step3 = KGRelationTuplePathGenerator(
            llm_serving=self.llm_serving,
            lang="en",
            k=2,
        )

        self.temporal_path_qa_generation_step4 = TKGTuplePathQAGeneration(
            llm_serving=self.llm_serving,
            lang="en",
            hop=2,
            qa_type="time_point",
            num_q=5,
        )

        self.qa_natural_eval_step5 = KGQANaturalEvaluator(
            llm_serving=self.llm_serving,
            lang="en",
        )

    def forward(self):
        self.temporal_tuple_extraction_step1.run(
            storage=self.storage.step(),
            input_key="text",
            output_key="tuple",
        )

        self.temporal_time_filter_step2.run(
            storage=self.storage.step(),
            input_key="tuple",
            output_key="filtered_tuple",
            query_time_start="Q1 2021",
            query_time_end="2023",
        )

        self.path_generation_step3.run(
            storage=self.storage.step(),
            input_key="tuple",
            output_key_meta="hop_paths",
        )

        self.temporal_path_qa_generation_step4.run(
            storage=self.storage.step(),
            input_key_meta="hop_paths",
            output_key_meta="QA_pairs",
        )

        self.qa_natural_eval_step5.run(
            storage=self.storage.step(),
            input_key="2_QA_pairs",
            output_key="naturalness_scores",
        )


if __name__ == "__main__":
    model = TKGQA_APIPipeline()
    model.forward()
