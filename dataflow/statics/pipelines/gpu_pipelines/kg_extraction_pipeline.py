from dataflow.utils.storage import FileStorage
from dataflow.serving import LocalModelLLMServing_vllm
# 如果你想用 SGLang，可以再引入：
# from dataflow.serving import LocalModelLLMServing_sglang

from dataflow.operators.general_kg.generate.kg_entity_extractor import (
    KGEntityExtraction,
    KGTripleExtraction,
    KGRelationTripleInference,
    KGTupleRemoveRepeated,
    KGTupleValidity
)


class KGExtractionPipeline_GPU:
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/KGExtractionPipeline.json",
            cache_path="./cache_kg_extraction",
            file_name_prefix="kg_extraction_pipeline",
            cache_type="jsonl",
        )

        # 使用本地 GPU 模型，不再调用远程 API
        self.llm_serving = LocalModelLLMServing_vllm(
            # 这里可以改成你的本地模型路径，或者 HuggingFace 模型名
            # 例如：
            # hf_model_name_or_path="/workspace/models/Qwen2.5-72B-Instruct-GPTQ-Int8"
            # hf_model_name_or_path="Qwen/Qwen2.5-7B-Instruct"
            hf_model_name_or_path="Qwen/Qwen2.5-7B-Instruct",

            # 如果只用 1 张 GPU，设为 1
            # 如果用 2 张 H100 跑 72B，可以设为 2
            vllm_tensor_parallel_size=1,

            # 最大生成 token 数，根据任务和显存调整
            vllm_max_tokens=8192,
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
    model = KGExtractionPipeline_GPU()
    model.forward()