from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # 修复：原拼写 "cskg_triple_adapbility_eval" 少了一个 't'，
    # 实际文件名为 `cskg_triple_adaptability_eval.py`。
    from .eval.cskg_triple_adaptability_eval import CSKGTripleAdaptabilityEvaluator
    from .eval.cskg_triple_rationale_eval import CSKGTripleRationaleEvaluator

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking

    cur_path = "dataflow/operators/commonsense_kg/"

    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/commonsense_kg/", _import_structure)
