from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .domain_kg_tuple_ontology_filtering import GeoKGTupleFilter
    from .finkg_get_ontology import FinKGGetBasicOntology, load_finkg_ontology
    from .geokg_get_ontology import GeoKGGetBasicOntology
    from .legalkg_get_ontology import LegalKGGetBasicOntology
    from .medkg_get_anatomy_biology_ontology import MedKGGetAnatomyBiologyOntology
    from .medkg_get_disease_diagnosis_ontology import MedKGGetDiseaseDiagnosisOntology
    from .medkg_get_drug_therapy_ontology import MedKGGetDrugTherapyOntology
    from .medkg_get_microbe_pathogen_ontology import MedKGGetMicrobePathogenOntology
    from .medkg_get_test_measurement_ontology import MedKGGetTestMeasurementOntology
    from .schokg_get_ontology import SchoKGGetOntology

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking

    cur_path = "dataflow/operators/domain_kg/utils/"

    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/domain_kg/utils/", _import_structure)
