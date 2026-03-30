from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .generate.finkg_get_ontology import FinKGGetBasicOntology
    from .generate.finkg_4tuple_extractor import FinKGTupleExtraction
    from .generate.finkg_table_tuple_extractor import FinKGTableTupleExtraction

    from .filter.finkg_4tuple_ontology_filtering import FinKGTupleFilter

    from .refine.finkg_relation_chain_inference import FinKGRelationChainInference

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking


    cur_path = "dataflow/operators/financial_kg/"


    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/financial_kg/", _import_structure)
