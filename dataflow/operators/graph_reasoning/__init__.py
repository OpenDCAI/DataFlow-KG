from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # reasoning
    from .generate.reasoning_path_search import KGReasoningPathSearch
    from .generate.reasoning_constrained_path_search import KGConstrainedPathSearch
    from .generate.reasoning_rel_generator import KGReasoningRelationGeneration

    from .eval.reasoning_path_length_eval import KGReasoningPathLengthEvaluator
    from .eval.reasoning_path_redundancy_eval import KGPathRedundancyEvaluator

    from .filter.reasoning_path_length_filtering import KGReasoningPathLengthFilter

    
else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking


    cur_path = "dataflow/operators/graph_reasoning/"


    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/graph_reasoning/", _import_structure)
