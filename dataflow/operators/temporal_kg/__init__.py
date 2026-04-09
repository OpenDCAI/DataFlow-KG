from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # graphrag
    from .generate.tkg_4tuple_extractor import TKGTupleExtraction
    from .generate.tkg_attri_4tuple_qa_generator import TKGAttributeQAGeneration
    from .generate.tkg_rel_4tuple_subgraph_qa_generator import TKGTupleSubgraphQAGeneration
    from .generate.tkg_rel_4tuple_path_qa_generator import TKGTuplePathQAGeneration
    from .generate.tkg_rel_4tuple_conversation_generator import TKGRelationTripletDialogueQAGeneration
    from .generate.tkg_4tuple_merge import TKGTupleMerger
    
    from .refinement.tkg_4tuple_disambiguation import TKGTupleDisambiguation

    from .filter.tkg_4tuple_time_sampling import TKGTupleTimeFilter

    from .eval.tkg_4tuple_time_summary import TKGTemporalStatistics

    
else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking


    cur_path = "dataflow/operators/temporal_kg/"


    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/temporal_kg/", _import_structure)
