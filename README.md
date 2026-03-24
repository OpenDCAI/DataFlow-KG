# DataFlow-KG

## 1、算子文件开发进度表（总计113，目标150+） ✅
| 图谱类型 | 文件夹名 | 算子个数 | 是否有未提交的算子 | 是否有未合并的算子 |
| --- | --- | --- | --- | --- |
| 通用图谱 | core_kg | 41 | 否 | 是 @zhengpin |
| 常识图谱 | commensense_kg | 8 | 否 | 否 |
| 图推断 | graph_reasoning | 7 | 否 | 否 |
| 图RAG | graph_rag | 9 | 否 | 否 |
| 超关系图谱 | hyper_relation_kg | 9 | 是 @wanpeng | 否 |
| 多模态图谱 | multi_model_kg | 8 | 否 | 否 |
| 时序图谱 | temporal_kg | 9 | 否 | 否 |
| sparql图谱 | sparql_kg | 待定 | 是 @xuemeng | 是 @zhengpin |
| 地理领域图谱 | geospatial_kg | 13 | 否 | 否 |
| 法律领域图谱 | legal_kg | 9 | 否 | 否 |

## 2、待开发的算子文件进度表 ❓
| 图谱类型 | 文件名夹 | 是否已有对应文件夹 | 算子个数 | 负责同学 | 
| --- | --- | --- | --- | --- |
| sparql图谱 | sparql_kg | 是 |待定 | @xuemeng |
| 超关系图谱 | hyper_relation_kg | 是 |3 | @wanpeng |
| 医学领域图谱 (考虑细分子领域) | 待定 | 否 |待定 | @wanpeng @jinke|
| 学者领域图谱 | scholar_kg | 否 | 待定 | @zhengpin |

## 3、DataFlow-KG算子开发负责表 📅
| 成员 | 已合并算子个数 |已合并算子文件 | 已填写开发文档个数 |
| --- | --- | --- | --- |
| @runhao | 9 | tkg | 9 |
| @wanpeng | 11 | kg_rel_triple_consistency_eval.py, kg_rel_triple_topology_eval.py, kg_triple_hallucination_eval.py, mmkg | 0|
| @xuemeng | 待定 | 尽快合并 @zhengpin | 0 |
| @jinke | 1 | geokg_entity_link2database.py | 0 |
| @zhengpin | 剩余 | -  | - | 
