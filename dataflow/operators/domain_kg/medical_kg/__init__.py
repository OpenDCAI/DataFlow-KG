from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # filter
    from .filter.medkg_triple_metapath_sampling import MedKGMetaPathGenerator

    # generate
    from .generate.medkg_triple_drug_action_mechanism_discovery import MedKGTripleDrugActionMechanismDiscovery
    from .generate.medkg_triple_drug_repositioning_discovery import MedKGTripleDrugRepositioningDiscovery
    from .generate.medkg_triple_extractor import MedKGTripleExtraction

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking

    cur_path = "dataflow/operators/domain_kg/medical_kg/"

    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/domain_kg/medical_kg/", _import_structure)
