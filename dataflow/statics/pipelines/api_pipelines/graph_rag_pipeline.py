import os

from dataflow.core import LLMServingABC
from dataflow.operators.graph_rag import (
    KGGraphRAGGetAnswer,
    KGGraphRAGQueryExtraction,
    KGGraphRAGSubgraphRetrieval,
    KGRAGAnswerPlausibilityFilter,
    KGRAGQuestionDifficultyEvaluation,
    KGRAGQuestionPlausibilityEvaluation,
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


class GraphRAGPipeline(PipelineABC):
    """GraphRAG pipeline: question -> subgraph retrieval -> answer -> answer filtering."""

    def __init__(
        self,
        first_entry_file_name: str,
        llm_serving: LLMServingABC,
        cache_path: str = "./cache_local",
        file_name_prefix: str = "graph_rag_pipeline_step",
        cache_type: str = "json",
        lang: str = "en",
        hop: int = 1,
        plausibility_min_score: float = 0.95,
        plausibility_max_score: float = 1.0,
    ):
        super().__init__()
        if llm_serving is None:
            raise ValueError("llm_serving is required for GraphRAGPipeline")

        self.storage = FileStorage(
            first_entry_file_name=first_entry_file_name,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )
        self.plausibility_min_score = plausibility_min_score
        self.plausibility_max_score = plausibility_max_score

        self.query_extraction_step1 = KGGraphRAGQueryExtraction(
            llm_serving=llm_serving,
            lang=lang,
        )
        self.subgraph_retrieval_step2 = KGGraphRAGSubgraphRetrieval(hop=hop)
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
            merge_to_input=False
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
    input_file = os.environ.get(
        "DF_GRAPHRAG_INPUT_FILE",
        _default_example_file("GraphRAGPipeline", "input.json"),
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

    pipeline = GraphRAGPipeline(
        first_entry_file_name=input_file,
        llm_serving=llm_serving,
        cache_path="./cache_graph_rag",
        lang="en",
        hop=1,
    )
    pipeline.forward()
