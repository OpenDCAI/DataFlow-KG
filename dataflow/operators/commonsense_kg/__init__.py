from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # graphrag
    # from .cskg_triple_extractor import CSKGTripleExtraction
    # from .cskg_triple_concept_generalization import CSKGTripleConceptGeneralization
    # from .cskg_rel_triple_set_sampling import CSKGRelationTripleSetSampling
    # from .cskg_rel_triple_qa_generator import CSKGRelationTripleQAGeneration

    from .eval.cskg_triple_adapbility_eval import CSKGTripleAdapbilityEvaluator
    from .eval.cskg_triple_rationale_eval import CSKGTripleRationaleEvaluator
    
else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking


    cur_path = "dataflow/operators/commonsense_kg/"


    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/commonsense_kg/", _import_structure)
