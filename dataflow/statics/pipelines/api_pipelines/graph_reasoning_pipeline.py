import os

from dataflow.core import LLMServingABC
from dataflow.operators.graph_reasoning import (
    KGReasoningPathLengthEvaluator,
    KGReasoningPathLengthFilter,
    KGReasoningPathSearch,
    KGReasoningRelationGeneration,
)
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


class GraphReasoningPipeline(PipelineABC):
    """Graph reasoning pipeline: target pairs -> multi-hop paths -> inferred relations."""

    def __init__(
        self,
        first_entry_file_name: str,
        llm_serving: LLMServingABC,
        cache_path: str = "./cache_local",
        file_name_prefix: str = "graph_reasoning_pipeline_step",
        cache_type: str = "json",
        lang: str = "en",
        max_hop: int = 4,
        min_length: int = 2,
        max_length: int = 4,
        restrict_to_path_rel: bool = True,
    ):
        super().__init__()
        if llm_serving is None:
            raise ValueError("llm_serving is required for GraphReasoningPipeline")

        self.storage = FileStorage(
            first_entry_file_name=first_entry_file_name,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )

        self.path_search_step1 = KGReasoningPathSearch(max_hop=max_hop)
        self.path_length_step2 = KGReasoningPathLengthEvaluator()
        self.path_filter_step3 = KGReasoningPathLengthFilter(
            min_length=min_length,
            max_length=max_length,
        )
        self.relation_generation_step4 = KGReasoningRelationGeneration(
            llm_serving=llm_serving,
            restrict_to_path_rel=restrict_to_path_rel,
            lang=lang,
        )

    def forward(self):
        self.path_search_step1.run(
            storage=self.storage.step(),
            input_key="triplet",
            output_key="mpath",
        )
        self.path_length_step2.run(
            storage=self.storage.step(),
            input_key="mpath",
            output_key="mpath_length",
        )
        self.path_filter_step3.run(
            storage=self.storage.step(),
            mpath_key="mpath",
            length_key="mpath_length",
            output_path_key="filtered_mpath",
        )
        self.relation_generation_step4.run(
            storage=self.storage.step(),
            target_key="target_entity",
            path_key="filtered_mpath",
            output_key="inferred_triplets",
        )


if __name__ == "__main__":
    input_file = os.environ.get(
        "DF_GRAPH_REASONING_INPUT_FILE",
        _default_example_file("GraphReasoningPipeline", "input.json"),
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

    pipeline = GraphReasoningPipeline(
        first_entry_file_name=input_file,
        llm_serving=llm_serving,
        cache_path="./cache_graph_reasoning",
        lang="en",
        max_hop=4,
        min_length=2,
        max_length=3,
    )
    pipeline.forward()
