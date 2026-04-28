from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # filter
    from .filter.finkg_4tuple_ontology_filtering import FinKGTupleFilter

    # generate
    from .generate.finkg_4tuple_extractor import FinKGTupleExtraction
    from .generate.finkg_entity_risk_assessment import FinKGEntityRiskAssessment
    from .generate.finkg_event_impact_tracing import FinKGEventImpactTracing
    from .generate.finkg_investment_analysis import FinKGInvestmentAnalysis
    from .generate.finkg_marketaux_news_retriever import FinKGMarketauxNewsRetriever

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking

    cur_path = "dataflow/operators/domain_kg/financial_kg/"

    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/domain_kg/financial_kg/", _import_structure)
