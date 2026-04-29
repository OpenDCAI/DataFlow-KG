import os

from dataflow.operators.graph_rag import (
    KGGraphRAGGetAnswer,
    KGGraphRAGQueryExtraction,
    KGGraphRAGSubgraphRetrieval,
    KGRAGAnswerPlausibilityFilter,
    KGRAGQuestionDifficultyEvaluation,
    KGRAGQuestionPlausibilityEvaluation,
)
from dataflow.pipeline import PipelineABC
from dataflow.serving import LocalModelLLMServing_vllm
from dataflow.utils.storage import FileStorage


class GraphRAGPipeline(PipelineABC):
    """GraphRAG pipeline: question -> subgraph retrieval -> answer -> answer filtering."""

    def __init__(
        self,
        lang: str = "en",
        hop: int = 1,
        plausibility_min_score: float = 0.95,
        plausibility_max_score: float = 1.0,
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
            first_entry_file_name="../example_data/GraphRAGPipeline/input.json",
            cache_path="./graph_rag",
            file_name_prefix="graph_rag_pipeline_step",
            cache_type="json",
        )

        self.plausibility_min_score = plausibility_min_score
        self.plausibility_max_score = plausibility_max_score

        self.query_extraction_step1 = KGGraphRAGQueryExtraction(
            llm_serving=llm_serving,
            lang=lang,
        )

        self.subgraph_retrieval_step2 = KGGraphRAGSubgraphRetrieval(
            hop=hop,
        )

        self.answer_generation_step3 = KGGraphRAGGetAnswer(
            llm_serving=llm_serving,
            lang=lang,
        )

        self.question_difficulty_step4 = KGRAGQuestionDifficultyEvaluation(
            llm_serving=llm_serving,
            lang=lang,
        )

        self.answer_plausibility_step5 = KGRAGQuestionPlausibilityEvaluation(
            llm_serving=llm_serving,
            lang=lang,
        )

        self.answer_filter_step6 = KGRAGAnswerPlausibilityFilter(
            merge_to_input=False,
        )

    def forward(self):
        self.query_extraction_step1.run(
            storage=self.storage.step(),
            input_key="question",
            output_keys=["entities", "relations"],
        )

        self.subgraph_retrieval_step2.run(
            storage=self.storage.step(),
            output_key="subgraph_prompt",
        )

        self.answer_generation_step3.run(
            storage=self.storage.step(),
            input_keys=["question", "subgraph_prompt"],
            output_key="answer",
        )

        self.question_difficulty_step4.run(
            storage=self.storage.step(),
            question_key="question",
            output_key="question_difficulty",
        )

        self.answer_plausibility_step5.run(
            storage=self.storage.step(),
            question_key="question",
            answer_key="answer",
            output_key="question_plausibility_score",
        )

        self.answer_filter_step6.run(
            storage=self.storage.step(),
            input_key="answer",
            score_key="question_plausibility_score",
            output_key="filtered_answer",
            min_score=self.plausibility_min_score,
            max_score=self.plausibility_max_score,
        )


if __name__ == "__main__":
    pipeline = GraphRAGPipeline(
        lang="en",
        hop=1,
    )
    pipeline.forward()