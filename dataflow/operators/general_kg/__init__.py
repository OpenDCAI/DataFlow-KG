from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # eval
    from .eval.kg_qa_concise_eval import KGQAConcisenessEvaluator
    from .eval.kg_qa_correlation_eval import KGQACorrelationEvaluator
    from .eval.kg_qa_natural_eval import KGQANaturalEvaluator
    from .eval.kg_rel_triple_consistency_eval import KGRelationTripleConsistencyEvaluator
    from .eval.kg_rel_triple_nx_visual import KGRelationTripleVisualization
    from .eval.kg_rel_triple_strength_eval import KGRelationStrengthScoring
    from .eval.kg_rel_triple_topology_eval import KGRelationTripleTopologyEvaluator
    from .eval.kg_subgraph_connectivity_eval import KGSubgraphConnectivityEvaluator
    from .eval.kg_subgraph_consistence_eval import KGSubgraphConsistency
    from .eval.kg_subgraph_scale_eval import KGSubgraphScaleEvaluator

    # filter
    from .filter.kg_attri_tuple_sampling import KGAttributeTupleSampler
    from .filter.kg_entity_validation import KGEntityValidity
    from .filter.kg_qa_concise_filtering import KGQAConciseFilter
    from .filter.kg_qa_correlation_filtering import KGQACorrelationFilter
    from .filter.kg_qa_natural_filtering import KGQANaturalFilter
    from .filter.kg_rel_triple_strength_filtering import KGTripleStrengthFilter
    from .filter.kg_rel_tuple_path_sampling import KGRelationTuplePathGenerator
    from .filter.kg_rel_tuple_subgraph_sampling import KGEntityBasedSubgraphSampling
    from .filter.kg_subgraph_connectivity_filtering import KGSubgraphConnectivityFilter
    from .filter.kg_subgraph_consistence_filtering import KGSubgraphConsistenceFilter
    from .filter.kg_subgraph_scale_filtering import KGSubgraphScaleFilter
    from .filter.kg_tuple_remove_repeated import KGTupleRemoveRepeated
    from .filter.kg_tuple_validation import KGTupleValidity

    # generate
    from .generate.kg_attri_triple_qa_generator import KGAttributeTripleQAGeneration
    from .generate.kg_entity_extractor import KGEntityExtraction
    from .generate.kg_rel_triple_conversation_generator import KGRelationTripletDialogueQAGeneration
    from .generate.kg_rel_triple_inference import KGRelationTripleInference
    from .generate.kg_rel_triple_path_qa_generator import KGRelationTriplePathQAGeneration
    from .generate.kg_rel_triple_subgraph_qa_generator import KGRelationTripleSubgraphQAGeneration
    from .generate.kg_triple_extractor import KGTripleExtraction
    from .generate.kg_triple_merge import KGTripleMerger
    from .generate.kg_tuple2text import KGTupleTextGeneration

    # refinement
    from .refinement.kg_entity_alignment import KGGraphEntityAligner
    from .refinement.kg_entity_classification import KGEntityClassification
    from .refinement.kg_entity_disambiguation import KGEntityDisambiguation
    from .refinement.kg_entity_link2database import KGEntityLink2Database
    from .refinement.kg_entity_normalization import KGEntityNormalization
    from .refinement.kg_triple_disambiguation import KGTripleDisambiguation
    from .refinement.kg_tuple_normalization import KGTupleNormalization

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking

    cur_path = "dataflow/operators/general_kg/"

    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/general_kg/", _import_structure)
