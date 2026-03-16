from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # graphrag
    from .graphrag_query_extractor import KGGraphRAGQueryExtraction
    from .graphrag_prompt_generator import KGGraphRAGSubgraphRetrieval
    from .graphrag_get_answer import KGGraphRAGGetAnswer
    from .graphrag_evaluate import KGGraphRAGAnswerLLMEvaluation

    from .eval.graphrag_answer_plausibility_eval import KGRAGQuestionPlausibilityEvaluation
    from .eval.graphrag_answer_token_eval import KGRAGAnswerTokenCount
    from .eval.graphrag_question_difficulty_eval import KGRAGQuestionDifficultyEvaluation
    from .filter.graphrag_answer_plausibility_filtering import KGRAGAnswerPlausibilityFilter

    
else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking


    cur_path = "dataflow/operators/graph_rag/"


    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/graph_rag/", _import_structure)
