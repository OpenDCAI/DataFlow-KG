from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # graphrag
    from .hrkg_rel_triple_extractor import HRKGTripleExtraction

    
else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking


    cur_path = "dataflow/operators/hyper_relation_kg/"


    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/hyper_relation_kg/", _import_structure)
