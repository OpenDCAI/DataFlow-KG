from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # generate
    from .generate.schokg_query_reasoning import SchoKGQueryReasoningOperator
    from .generate.schokg_recommend import SchoKGRecommendOperator
    from .generate.schokg_triple_extractor import SchoKGTripleExtraction

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking

    cur_path = "dataflow/operators/domain_kg/scholar_kg/"

    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/domain_kg/scholar_kg/", _import_structure)
