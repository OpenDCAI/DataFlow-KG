from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # generate
    from .generate.geokg_4tuple_extractor import GeoKGTupleExtraction
    from .generate.geokg_event_extractor import GeoKGEventExtraction

    # filter
    from .filter.geokg_event_time_filtering import GeoKGEventTupleTimeFilter
    from .filter.geokg_event_location_filtering import GeoKGEventTupleLocationFilter
    from .filter.geokg_event_rationale_filtering import GeoKGEventRationaleFilter
    from .filter.geokg_event_consistence_filtering import GeoKGEventConsistenceFilter

    # refine
    from .refine.geokg_rel_4tuple_inference import GeoKGRelationInference
    from .refine.geokg_entity_link2database import GeoKGEntityLink2Database

    # eval
    from .eval.geokg_event_consistence_eval import GeoKGEventConsistenceEvaluator
    from .eval.geokg_event_rationale_eval import GeoKGEventRationaleEvaluator
    from .eval.geokg_event_summary import GeoKGTupleAttributeFrequencyEvaluator

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking

    cur_path = "dataflow/operators/domain_kg/geospatial_kg/"

    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(
        __name__,
        "dataflow/operators/domain_kg/geospatial_kg/",
        _import_structure,
    )
