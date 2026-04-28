from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # eval
    from .eval.cskg_triple_adaptability_eval import CSKGTripleAdaptabilityEvaluator
    from .eval.cskg_triple_rationale_eval import CSKGTripleRationaleEvaluator

    # filter
    from .filter.cskg_rel_triple_set_sampling import CSKGRelationTripleSetSampling
    from .filter.cskg_triple_adaptability_filtering import CSKGTripleAdapbilityFilter
    from .filter.cskg_triple_rationale_filtering import CSKGTripleRationaleFilter

    # generate
    from .generate.cskg_rel_triple_qa_generator import CSKGRelationTripleQAGeneration
    from .generate.cskg_triple_extractor import CSKGTripleExtraction

    # refine
    from .refine.cskg_triple_concept_generalization import CSKGTripleConceptGeneralization

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking

    cur_path = "dataflow/operators/commonsense_kg/"

    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/commonsense_kg/", _import_structure)
