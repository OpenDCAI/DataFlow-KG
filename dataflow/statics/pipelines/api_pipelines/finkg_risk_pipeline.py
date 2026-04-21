import os

from dataflow.core import LLMServingABC
from dataflow.operators.domain_kg.financial_kg.filter.finkg_4tuple_ontology_filtering import (
    FinKGTupleFilter,
)
from dataflow.operators.domain_kg.financial_kg.generate.finkg_4tuple_extractor import (
    FinKGTupleExtraction,
)
from dataflow.operators.domain_kg.financial_kg.refine.finkg_entity_risk_assessment import (
    FinKGEntityRiskAssessment,
)
from dataflow.operators.domain_kg.utils.finkg_get_ontology import load_finkg_ontology
from dataflow.pipeline import PipelineABC
from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage


def _default_example_file(folder: str, file_name: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(os.path.join(script_dir, "..", "example_data", folder, file_name)),
        os.path.abspath(os.path.join(script_dir, "..", "..", "..", "example", folder, file_name)),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


class FinKGRiskPipeline(PipelineABC):
    """Financial KG pipeline: raw text -> tuples -> filtered tuples -> risk answer.

    Required dataset columns:
    - `raw_chunk`: source financial text
    - `target_entity`: entity whose risk should be assessed
    """

    def __init__(
        self,
        first_entry_file_name: str,
        llm_serving: LLMServingABC,
        cache_path: str = "./cache_local",
        file_name_prefix: str = "finkg_risk_pipeline_step",
        cache_type: str = "jsonl",
        lang: str = "en",
        triple_type: str = "relation",
        target_ontology: str = "Corporation",
    ):
        super().__init__()
        if llm_serving is None:
            raise ValueError("llm_serving is required for FinKGRiskPipeline")

        self.storage = FileStorage(
            first_entry_file_name=first_entry_file_name,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )
        self.ontology = load_finkg_ontology()
        self.target_ontology = target_ontology

        self.tuple_extraction_step1 = FinKGTupleExtraction(
            llm_serving=llm_serving,
            triple_type=triple_type,
            lang=lang,
        )
        self.tuple_filter_step2 = FinKGTupleFilter()
        self.risk_answer_step3 = FinKGEntityRiskAssessment(
            llm_serving=llm_serving,
            lang=lang,
        )

    def forward(self):
        self.tuple_extraction_step1.run(
            storage=self.storage.step(),
            ontology_lists=self.ontology,
            input_key="raw_chunk",
            input_key_meta=None,
            output_key="tuple",
            output_key_meta="entity_class",
        )
        self.tuple_filter_step2.run(
            storage=self.storage.step(),
            ontology_lists=self.ontology,
            input_key_tuple="tuple",
            input_key_class="entity_class",
            output_key="tuple",
            input_key_meta=None,
            target_ontology=self.target_ontology,
        )
        self.risk_answer_step3.run(
            storage=self.storage.step(),
            input_key="tuple",
            output_key="risk_answer",
            output_key_score="risk_score",
        )

    def _compiled_forward(self, resume_step: int = 0):
        for idx, op_node in enumerate(self.op_nodes_list):
            if idx - 1 < resume_step:
                continue
            if op_node.op_obj is None:
                continue

            if op_node.llm_serving is not None:
                if (
                    self.active_llm_serving
                    and self.active_llm_serving is not op_node.llm_serving
                ):
                    self.active_llm_serving.cleanup()
                self.active_llm_serving = op_node.llm_serving

            op_node.op_obj.run(storage=op_node.storage, **op_node.kwargs)

            if op_node.llm_serving is not None:
                self.llm_serving_counter[self.active_llm_serving] -= 1
                if self.llm_serving_counter[self.active_llm_serving] == 0:
                    self.active_llm_serving.cleanup()
                    self.active_llm_serving = None


if __name__ == "__main__":
    input_file = os.environ.get(
        "DF_FINKG_INPUT_FILE",
        _default_example_file("FinKGRiskPipeline", "input.json"),
    )

    llm_serving = APILLMServing_request(
        api_url=os.environ.get(
            "DF_API_URL",
            "https://api.openai.com/v1/chat/completions",
        ),
        key_name_of_api_key="DF_API_KEY",
        model_name=os.environ.get("DF_LLM_MODEL", "gpt-4o-mini"),
        max_workers=8,
        temperature=0.0,
    )

    pipeline = FinKGRiskPipeline(
        first_entry_file_name=input_file,
        llm_serving=llm_serving,
        cache_path="./cache_finkg_risk",
        lang="en",
        target_ontology="Corporation",
    )
    pipeline.forward()
