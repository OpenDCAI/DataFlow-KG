import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC


@OPERATOR_REGISTRY.register()
class LegalKGGetBasicOntology(OperatorABC):

    def __init__(self, lang: str = "zh"):
        self.logger = get_logger()
        self.lang = lang.lower()  # "zh" or "en"

    @staticmethod
    def get_desc(lang: str = "zh") -> tuple:
        if lang.lower() == "zh":
            return (
                "LegalKGGetBasicOntology：通用法律知识图谱本体",
                "覆盖刑事/民事/行政案件，支持事件建模",
                "输出: entity_type, relation_type, attribute_type"
            )
        else:
            return (
                "LegalKGGetBasicOntology: General Legal Knowledge Graph Ontology",
                "Supports criminal, civil, administrative cases and event modeling",
                "Output: entity_type, relation_type, attribute_type"
            )

    # =========================
    # Entity Ontology
    # =========================
    def load_entity_types(self):
        if self.lang == "zh":
            entity_types = {
                "主体": ["自然人", "法人", "组织机构"],
                "诉讼角色": ["原告", "被告", "受害人", "上诉人", "被上诉人", "检察官", "法官", "律师", "第三方"],
                "法律机构": ["法院", "检察院", "公安机关", "行政机关", "看守所", "监狱", "律师事务所"],
                "案件": ["刑事案件", "民事案件", "行政案件", "执行案件"],
                "法律概念": ["罪名", "诉讼请求", "法律权利", "法律义务", "法律条文", "证据", "指控", "判决", "裁定"],
                "法律事件": ["违法行为", "民事行为", "程序行为", "合同行为", "侵权行为"],
                "客体": ["财产", "金钱", "物品", "服务", "知识产权"],
                "时间": ["日期时间"],
                "地点": ["地点", "城市", "区域"]
            }
        else:
            entity_types = {
                "Actor": ["NaturalPerson", "LegalPerson", "Organization"],
                "LitigationRole": ["Plaintiff", "Defendant", "Victim", "Appellant", "Appellee", "Prosecutor", "Judge", "Lawyer", "ThirdParty"],
                "LegalInstitution": ["Court", "Procuratorate", "PublicSecurity", "AdministrativeAgency", "DetentionCenter", "Prison", "LawFirm"],
                "Case": ["CriminalCase", "CivilCase", "AdministrativeCase", "EnforcementCase"],
                "LegalConcept": ["Crime", "CauseOfAction", "LegalRight", "LegalObligation", "LawArticle", "Evidence", "Charge", "Judgment", "Ruling"],
                "LegalEvent": ["IllegalAct", "CivilAct", "ProceduralAct", "ContractAct", "TortAct"],
                "Object": ["Property", "Money", "Goods", "Service", "IntellectualProperty"],
                "Time": ["DateTime"],
                "Location": ["Place", "City", "Region"]
            }
        return entity_types

    # =========================
    # Relation Ontology
    # =========================
    def load_relation_types(self):
        if self.lang == "zh":
            relation_types = {
                "案件关系": ["包含当事人", "由…办理", "由…提起", "上诉至", "属于案件类型"],
                "角色关系": ["担任角色", "代表", "辩护", "起诉"],
                "行为关系": ["实施", "针对", "影响", "导致"],
                "法律认定": ["构成", "违反", "依据", "支持"],
                "司法关系": ["由…判决", "判处", "支付", "确认", "驳回", "撤销"],
                "程序关系": ["立案", "受理", "开庭", "羁押", "逮捕", "取保候审", "上诉", "执行"],
                "权利义务": ["享有权利", "承担义务", "违反", "履行"],
                "时间关系": ["发生于", "开始于", "结束于"],
                "空间关系": ["发生在", "位于"]
            }
        else:
            relation_types = {
                "CaseRelation": ["has_party","handled_by","filed_by","appealed_to","belongs_to_case_type"],
                "RoleRelation": ["plays_role","represents","defends","prosecutes"],
                "ActionRelation": ["commits","against","affects","results_in"],
                "LegalDetermination": ["constitutes","violates","based_on","supported_by"],
                "JudicialRelation": ["judged_by","sentenced_to","ordered_to_pay","recognized","dismissed","revoked"],
                "ProceduralRelation": ["filed","accepted","heard","detained","arrested","released_on_bail","appealed","executed"],
                "RightsObligations": ["has_right","has_obligation","breaches","fulfills"],
                "TemporalRelation": ["occurs_at","starts_at","ends_at"],
                "SpatialRelation": ["occurs_in","located_in"]
            }
        return relation_types

    # =========================
    # Attribute Ontology
    # =========================
    def load_attribute_types(self):
        if self.lang == "zh":
            attribute_types = {
                "主体属性": ["姓名","性别","出生日期","身份证号","地址","职业"],
                "案件属性": ["案号","案件类型","审理程序","法院等级"],
                "事件属性": ["金额","价值","方式","意图","后果"],
                "证据属性": ["证据类型","来源","可信度"],
                "判决属性": ["刑期","罚金","赔偿金额","缓刑","判决结果","生效日期"],
                "程序属性": ["阶段","状态"],
                "时间属性": ["时间戳","持续时间"]
            }
        else:
            attribute_types = {
                "ActorAttribute": ["name","gender","birth_date","id_number","address","occupation"],
                "CaseAttribute": ["case_id","case_type","trial_procedure","court_level"],
                "EventAttribute": ["amount","value","means","intent","consequence"],
                "EvidenceAttribute": ["evidence_type","source","credibility"],
                "JudgmentAttribute": ["sentence_length","fine_amount","compensation_amount","probation","judgment_result","effective_date"],
                "ProcedureAttribute": ["stage","status"],
                "TemporalAttribute": ["timestamp","duration"]
            }
        return attribute_types

    # =========================
    # Run
    # =========================
    def run(self, storage: DataFlowStorage = None):
        self.logger.info("Loading LegalKG ontology")

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
            file_path="./.cache/api/legal_ontology.json",
            use_current_step=False
        )

        self.logger.info(f"Ontology saved to {output_file}")

        return ["entity_type","relation_type","attribute_type"]