from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .generate.geokg_get_ontology import GeoKGGetBasicOntology
    from .generate.geokg_4tuple_extractor import GeoKGTupleExtraction
    from .generate.geokg_event_extractor import GeoKGEventExtraction

    from .filter.geokg_4tuple_ontology_filtering import GeoKGTupleFilter
    from .filter.geokg_event_time_filtering import GeoKGEventTupleTimeFilter
    from .filter.geokg_event_location_filtering import GeoKGEventTupleLocationFilter

    from .refine.geokg_rel_4tuple_inference import GeoKGRelationInference
    from .refine.geokg_entity_link2database import GeoKGEntityLink2Database

    from .eval.geokg_event_consistence_eval import GeoKGEventConsistenceEvaluator
    from .eval.geokg_event_rationale_eval import GeoKGEventRationaleEvaluator
    from .eval.geokg_event_summary import GeoKGTupleAttributeFrequencyEvaluator

    
else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking


    cur_path = "dataflow/operators/geospatial_kg/"


    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/geospatial_kg/", _import_structure)
