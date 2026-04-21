# -*- coding: utf-8 -*-
"""
====================================
DataFlow-KG: FinKGGetBasicOntology
====================================

License:
    MIT License
"""

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage, FileStorage


def load_finkg_ontology(
    ontology_lists: Optional[Any] = None,
    input_key_meta: str = "finkg_ontology",
) -> Dict[str, Any]:
    if ontology_lists is not None:
        ontology = ontology_lists[0] if isinstance(ontology_lists, list) else ontology_lists
        return {
            "entity_type": ontology.get("entity_type", {}),
            "relation_type": ontology.get("relation_type", {}),
            "attribute_type": ontology.get("attribute_type", {}),
        }

    ontology_path = Path(f"./.cache/api/{input_key_meta}.json")
    if ontology_path.exists():
        storage_meta = FileStorage(first_entry_file_name="", cache_type="json")
        ontology_df = storage_meta.read(
            file_path=str(ontology_path),
            output_type="dataframe",
        )
        row = ontology_df.iloc[0]
        return {
            "entity_type": row["entity_type"],
            "relation_type": row["relation_type"],
            "attribute_type": row.get("attribute_type", {}),
        }

    ontology_loader = FinKGGetBasicOntology()
    return {
        "entity_type": ontology_loader.load_entity_types(),
        "relation_type": ontology_loader.load_relation_types(),
        "attribute_type": ontology_loader.load_attribute_types(),
    }


@OPERATOR_REGISTRY.register()
class FinKGGetBasicOntology(OperatorABC):

    def __init__(self):
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "FinKGGetBasicOntology 用于加载金融知识图谱基础本体。",
                "输入: 无; 输出: entity_type + relation_type + attribute_type",
            )
        return (
            "FinKGGetBasicOntology is used to load the basic ontology for Financial KG.",
            "Input: none; Output: entity_type + relation_type + attribute_type",
        )

    # =========================
    # Entity Ontology
    # =========================
    # Design rationale (FIBO-aligned):
    #   Agent  = WHO (classified by legal structure)
    #   Role   = WHAT THEY DO (classified by functional role)
    #   A single Agent can play multiple Roles via "plays_role" relation.
    #   Ref: FIBO BE domain — Agent / FunctionalEntity separation.

    def load_entity_types(self):

        entity_types = {

            # Agent: 按法律结构分类（FIBO BE domain）
            "Agent": [
                "NaturalPerson",
                "Corporation", "Partnership", "Trust",
                "GovernmentEntity", "NonProfitOrganization",
                "AutomatedSystem"
            ],

            # Role: 按功能角色分类（FIBO FunctionalEntity）
            # 同一 Agent 可有多个 Role
            "Role": [
                "IssuerRole", "InvestorRole",
                "LenderRole", "BorrowerRole",
                "UnderwriterRole", "BrokerRole", "DealerRole",
                "CustodianRole", "GuarantorRole",
                "ObligorRole", "CounterpartyRole",
                "ManagerRole", "RegulatorRole"
            ],

            # FinancialInstrument: 仅叶子类型（FIBO FBC/SEC 分层）
            # 父类如 DebtInstrument / DerivativeInstrument / Security
            # 不作为抽取标签，仅用于上层分类注释。
            "FinancialInstrument": [
                "CashInstrument", "CommodityInstrument", "CurrencyInstrument",
                "GovernmentBond", "CorporateBond", "ConvertibleBond",
                "Future", "Option", "Swap", "Entitlement",
                "EquityInstrument", "Fund"
            ],

            # Agreement: 合同/协议（FIBO Situation → Agreement）
            "Agreement": [
                "WrittenContract", "MasterAgreement",
                "LoanAgreement", "GuaranteeAgreement",
                "CollateralAgreement", "RepoAgreement",
                "CreditFacilityAgreement"
            ],

            # Occurrence: 事件/交易（FIBO Occurrence）
            "Occurrence": [
                "Transaction", "Trade", "Payment", "Settlement",
                "CorporateAction", "DefaultEvent",
                "RegulatoryAction", "MonetaryPolicyDecision",
                "LifecycleEvent"
            ],

            # Market: 市场类型
            "Market": [
                "ExchangeMarket", "OTCMarket", "MoneyMarket",
                "BondMarket", "EquityMarket",
                "ForeignExchangeMarket", "CommodityMarket",
                "DerivativesMarket"
            ],

            # IndexBenchmark: 指数与基准利率
            "IndexBenchmark": [
                "EquityIndex", "BondIndex", "CommodityIndex",
                "VolatilityIndex", "SectorIndex",
                "ReferenceRate", "CreditIndex"
            ],

            # ReferenceEntity: 参考实体
            "ReferenceEntity": [
                "Country", "Jurisdiction", "Currency",
                "Sector", "Industry", "Commodity",
                "EconomicIndicator", "CreditRatingScale",
                "Regulation"
            ]
        }

        return entity_types

    # =========================
    # Relation Ontology
    # =========================

    def load_relation_types(self):

        relation_types = {

            # Agent ↔ Role 连接（FIBO 核心：同一主体多角色）
            "IdentityRoleRelation": [
                "plays_role", "role_of",
                "acts_for", "appointed_as"
            ],

            # 治理/授权关系（从 IdentityRoleRelation 拆出）
            "GovernanceRelation": [
                "managed_by", "authorized_by",
                "reports_to", "audited_by"
            ],

            "OwnershipControlRelation": [
                "owns", "controls", "beneficially_owns",
                "major_shareholder_of", "subsidiary_of", "parent_of"
            ],

            # 合同/协议相关
            "ContractRelation": [
                "party_to", "counterparty_of", "obligor_in",
                "guarantor_of", "secured_by",
                "collateral_for", "governed_by"
            ],

            "IssuanceFinancingRelation": [
                "issues", "underwrites",
                "borrows_from", "lends_to",
                "guarantees", "repays_to"
            ],

            "TradingMarketRelation": [
                "listed_on", "traded_on", "quoted_in",
                "market_maker_for", "cleared_by",
                "settled_via", "delisted_from"
            ],

            # 工具之间的结构关系
            "InstrumentStructureRelation": [
                "classified_as", "underlying_of", "derives_from",
                "convertible_to", "benchmarked_to",
                "constituent_of", "tracks"
            ],

            "RiskRegulatoryRelation": [
                "regulated_by", "licensed_by", "complies_with",
                "violates", "fined_by", "sanctioned_by",
                "downgraded_by", "upgraded_by",
                "defaults_on", "hedges_with"
            ],

            # 事件关联
            "OccurrenceRelation": [
                "executes", "participates_in",
                "announced_by", "triggers",
                "results_in", "affects"
            ]
        }

        return relation_types

    # =========================
    # Attribute Ontology
    # =========================

    def load_attribute_types(self):

        attribute_types = {

            # 标识符（FIBO Designation）
            "DesignationAttribute": [
                "lei", "isin", "ticker",
                "cusip", "account_number", "trade_id"
            ],

            # 主体属性（对应 Agent）
            "AgentAttribute": [
                "legal_name", "legal_form", "incorporation_type",
                "domicile", "jurisdiction",
                "headquarters_location", "ownership_structure"
            ],

            # 角色属性（对应 Role）
            "RoleAttribute": [
                "role_status", "authorization_scope",
                "mandate_type", "fiduciary_level",
                "counterparty_tier"
            ],

            "FinancialStatementAttribute": [
                "revenue", "net_income", "total_assets",
                "total_liabilities", "total_equity",
                "cash_flow_from_operations", "eps"
            ],

            # 工具属性
            "InstrumentAttribute": [
                "notional_amount", "face_value", "coupon_rate",
                "yield_to_maturity", "strike_price",
                "maturity_tenor", "contract_size",
                "underlying_asset_type", "leverage_factor"
            ],

            "MarketAttribute": [
                "close_price", "volume", "price_change_pct",
                "volatility", "bid_ask_spread", "liquidity_score"
            ],

            "RiskComplianceAttribute": [
                "credit_rating", "probability_of_default",
                "value_at_risk", "capital_adequacy_ratio",
                "non_performing_loan_ratio", "leverage_ratio",
                "liquidity_coverage_ratio", "expected_loss"
            ],

            "BenchmarkAttribute": [
                "index_level", "tracking_error",
                "beta", "duration", "spread", "turnover_ratio"
            ]
        }

        return attribute_types

    # =========================
    # Run
    # =========================

    def run(
        self,
        storage: DataFlowStorage = None
    ):

        self.logger.info("Loading FinKG ontology")

        entity_types = self.load_entity_types()
        relation_types = self.load_relation_types()
        attribute_types = self.load_attribute_types()

        dataframe = pd.DataFrame({
            "entity_type": [entity_types],
            "relation_type": [relation_types],
            "attribute_type": [attribute_types],
        })

        output_file = storage.write(
            dataframe,
            file_path="./.cache/api/finkg_ontology.json",
            use_current_step=False
        )

        self.logger.info(f"Ontology saved to {output_file}")

        return ["entity_type", "relation_type", "attribute_type"]
