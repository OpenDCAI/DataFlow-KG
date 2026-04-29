import os

from dataflow.serving import LocalModelLLMServing_vllm
from dataflow.utils.storage import FileStorage

from dataflow.operators.general_kg.eval.kg_qa_natural_eval import KGQANaturalEvaluator
from dataflow.operators.hyper_relation_kg import HRKGRelationTripleAttributeFilter
from dataflow.operators.hyper_relation_kg import (
    HRKGTripleExtraction,
)
from dataflow.operators.hyper_relation_kg import (
    HRKGRelationTripleSubgraphQAGeneration,
)


class HyperKGQA_GPUPipeline:
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/HyperRelationKGPipeline/input.json",
            cache_path="./hyper_kg",
            file_name_prefix="hyper_kg_qa_pipeline",
            cache_type="json",
        )

        # 使用本地 GPU 模型，不再调用远程 API
        self.llm_serving = LocalModelLLMServing_vllm(
            # 可以是 HuggingFace 模型名，也可以是本地模型路径
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
    model = HyperKGQA_GPUPipeline()
    model.forward()