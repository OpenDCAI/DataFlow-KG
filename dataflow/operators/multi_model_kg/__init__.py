from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # generate
    from .generate.mmkg_visual_triple_extractor import MMKGVisualTripleExtraction
    from .generate.mmkg_visual_triple_subgraph_qa_generator import MMKGSubgraphBaseQAGeneration
    from .generate.mmkg_visual_triple_path_qa_generator import MMKGPathBaseQAGeneration

    # refine
    from .refine.mmkg_entity_link2database import MMKGImgDictLink2WikiSimple
    from .refine.mmkg_entity_link2img import MMKGEntityLink2ImgUrl

    # filter
    from .filter.mmkg_visual_triple_subgraph_sampling import MMKGEntityBasedSubgraphSampling
    from .filter.mmkg_visual_triple_path_sampling import MMKGRelationTuplePathGenerator

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking

    cur_path = "dataflow/operators/multi_model_kg/"

    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/multi_model_kg/", _import_structure)
