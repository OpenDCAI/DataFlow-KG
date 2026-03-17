from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .generate.legalkg_get_ontology import LegalKGGetBasicOntology
    from .generate.legalkg_triple_extractor import LegalKGTupleExtraction

    
else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking


    cur_path = "dataflow/operators/legal_kg/"


    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/legal_kg/", _import_structure)
