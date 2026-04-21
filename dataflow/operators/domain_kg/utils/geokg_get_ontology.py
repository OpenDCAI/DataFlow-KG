import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC


@OPERATOR_REGISTRY.register()
class GeoKGGetBasicOntology(OperatorABC):

    def __init__(self, lang: str = "en"):
        self.logger = get_logger()
        self.lang = lang.lower()

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "GeoKGGetBasicOntology 用于加载地理知识图谱基础本体。",
                "包含实体类型、关系类型、属性类型及时空类型。",
                "输出: entity_type, relation_type, attribute_type, temporal_type"
            )
        return (
            "GeoKGGetBasicOntology loads the basic ontology for GeoKG.",
            "Includes entity types, relation types, attribute types, and temporal types.",
            "Output: entity_type, relation_type, attribute_type, temporal_type"
        )

    # =========================
    # Entity Ontology
    # =========================

    def load_entity_types(self):

        if self.lang == "zh":
            return {
                "自然地理要素": [
                    "山脉","火山","高原","山谷",
                    "河流","湖泊","海洋","冰川","沙漠","森林"
                ],
                "行政区划": [
                    "国家","州","省","地级市",
                    "城市","县","区","镇","村"
                ],
                "基础设施": [
                    "道路","高速公路","铁路","桥梁",
                    "隧道","机场","港口","大坝","运河"
                ]
            }

        # default: English
        return {
            "NaturalFeature": [
                "Mountain","Volcano","Plateau","Valley",
                "River","Lake","Ocean","Glacier","Desert","Forest"
            ],
            "AdministrativeRegion": [
                "Country","State","Province","Prefecture",
                "City","County","District","Town","Village"
            ],
            "Infrastructure": [
                "Road","Highway","Railway","Bridge",
                "Tunnel","Airport","Port","Dam","Canal"]
        }

    # =========================
    # Relation Ontology
    # =========================

    def load_relation_types(self):

        if self.lang == "zh":
            return {
                "空间关系": [
                    "位于","属于","邻接",
                    "接近","包含","在...之内"
                ],
                "拓扑关系": [
                    "相交","接触","穿越","不相交"
                ],
                "水文关系": [
                    "流经","流入","发源于","支流属于"
                ],
                "行政关系": [
                    "首都","治理","管辖","属于区域"
                ],
                "基础设施关系": [
                    "通过连接","服务于","可通过到达"
                ],
                "时间关系": [
                    "存在于","建造于","成立于","废止于"
                ]
            }

        return {
            "SpatialRelation": [
                "located_in","part_of","adjacent_to",
                "near","contains","within"
            ],
            "TopologicalRelation": [
                "intersects","touches","crosses","disjoint"
            ],
            "HydrologicalRelation": [
                "flows_through","flows_into","originates_from","tributary_of"
            ],
            "AdministrativeRelation": [
                "capital_of","governs","administers","belongs_to_region"
            ],
            "InfrastructureRelation": [
                "connected_by","served_by","accessible_via"
            ],
            "TemporalRelation": [
                "existed_during","built_in","founded_in","abolished_in"
            ]
        }

    # =========================
    # Attribute Ontology
    # =========================

    def load_attribute_types(self):

        if self.lang == "zh":
            return {
                "空间属性": [
                    "纬度","经度","海拔",
                    "面积","长度","宽度","深度"
                ],
                "行政属性": [
                    "人口","人口密度",
                    "邮政编码","行政编码"
                ],
                "环境属性": [
                    "气候类型","平均气温",
                    "年降水量","植被类型"
                ],
                "经济属性": [
                    "GDP","人均GDP","主要产业"
                ],
                "时间属性": [
                    "建立日期","建设日期","解散日期",
                    "历史时期","观测时间"
                ]
            }

        return {
            "SpatialAttribute": [
                "latitude","longitude","elevation",
                "area","length","width","depth"
            ],
            "AdministrativeAttribute": [
                "population","population_density",
                "postal_code","administrative_code"
            ],
            "EnvironmentalAttribute": [
                "climate_type","average_temperature",
                "annual_rainfall","vegetation_type"
            ],
            "EconomicAttribute": [
                "GDP","GDP_per_capita","major_industry"
            ],
            "TemporalAttribute": [
                "established_date","construction_date","dissolution_date",
                "historical_period","observation_time"
            ]
        }

    # =========================
    # Run
    # =========================

    def run(self, 
        storage: DataFlowStorage = None
        ):

        self.logger.info(f"Loading GeoKG ontology (lang={self.lang})")

        entity_types = self.load_entity_types()
        relation_types = self.load_relation_types()
        attribute_types = self.load_attribute_types()

        dataframe = pd.DataFrame({
            "entity_type": [entity_types],
            "relation_type": [relation_types],
            "attribute_type": [attribute_types],
        })

        file_path = f"./.cache/api/ontology_{self.lang}.json"

        output_file = storage.write(
            dataframe, 
            file_path=file_path, 
            use_current_step=False
        )

        self.logger.info(f"Ontology saved to {output_file}")

        return ["entity_type","relation_type","attribute_type","temporal_type"]
