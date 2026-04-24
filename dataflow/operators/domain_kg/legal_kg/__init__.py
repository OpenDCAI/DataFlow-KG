from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dataflow.operators.domain_kg.utils.legalkg_get_ontology import LegalKGGetBasicOntology
    from .generate.legalkg_triple_extractor import LegalKGTupleExtraction
    from .generate.legalkg_case_judgement_generator import LegalKGJudgementPrediction

    from .filter.legalkg_case_similarity_filtering import LegalKGCaseSimilarityFilter

    from .eval.legalkg_case_similarity_eval import LegalKGCaseSummarySimilarity

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking

    cur_path = "dataflow/operators/legal_kg/"

    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/legal_kg/", _import_structure)
