import os

from dataflow.operators.graph_reasoning import (
    KGReasoningPathLengthEvaluator,
    KGReasoningPathLengthFilter,
    KGReasoningPathSearch,
    KGReasoningRelationGeneration,
)
from dataflow.pipeline import PipelineABC
from dataflow.serving import LocalModelLLMServing_vllm
from dataflow.utils.storage import FileStorage


class GraphReasoningPipeline(PipelineABC):
    """Graph reasoning pipeline: target pairs -> multi-hop paths -> inferred relations."""

    def __init__(
        self,
        lang: str = "en",
        max_hop: int = 4,
        min_length: int = 2,
        max_length: int = 4,
        restrict_to_path_rel: bool = True,
    ):
        super().__init__()

        # 使用本地 GPU 模型，不再调用远程 API
        llm_serving = LocalModelLLMServing_vllm(
            # 可以写 HuggingFace 模型名，也可以写本地模型路径
            # 例如：
            # hf_model_name_or_path="Qwen/Qwen2.5-7B-Instruct"
            # hf_model_name_or_path="/workspace/models/Qwen2.5-7B-Instruct"
            # hf_model_name_or_path="/workspace/models/Qwen2.5-72B-Instruct"
            hf_model_name_or_path=os.environ.get(
                "DF_LOCAL_MODEL",
                "Qwen/Qwen2.5-7B-Instruct",
            ),

            # 1 张 GPU 用 1；2 张 H100 跑 72B 可以设为 2
            vllm_tensor_parallel_size=int(
                os.environ.get("DF_VLLM_TP_SIZE", "1")
            ),

            # 最大生成 token 数
            vllm_max_tokens=int(
                os.environ.get("DF_VLLM_MAX_TOKENS", "8192")
            ),
        )

        self.storage = FileStorage(
            first_entry_file_name="../example_data/GraphReasoningPipeline/input.json",
            cache_path="./graph_reasoning",
            file_name_prefix="graph_reasoning_pipeline_step",
            cache_type="json",
        )

        self.path_search_step1 = KGReasoningPathSearch(
            max_hop=max_hop,
        )

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
            input_key="triple",
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
            output_key="inferred_triples",
        )


if __name__ == "__main__":
    pipeline = GraphReasoningPipeline(
        lang="en",
        max_hop=4,
        min_length=2,
        max_length=3,
    )
    pipeline.forward()