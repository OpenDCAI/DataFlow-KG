# DataFlow-KG

## 1、算子文件开发进度表（总计101） ✅
| 图谱类型 | 文件名 | 算子个数 | 是否有未提交的算子 | 是否有未合并的算子 |
| --- | --- | --- | --- | --- |
| 通用图谱 | core_kg | 41 | 否 | 是 @zhengpin |
| 常识图谱 | commensense_kg | 8 | 否 | 否 |
| 图推断 | graph_reasoning | 7 | 否 | 否 |
| 图RAG | graph_rag | 8 | 否 | 否 |
| 超关系图谱 | hyper_relation_kg | 9 | 是 @runhao | 否 |
| 多模态图谱 | multi_model_kg | 8 | 否 | 否 @zhengpin |
| 时序图谱 | temporal_kg | 9 | 否 | 否 |
| sparql图谱 | sparql_kg | 待定 | 是 @xuemeng | 是 @zhengpin |
| 地理领域图谱 | geospatial_kg | 11 | 否 | 否 |
| 法律领域图谱 | legal_kg | 待定 | 是 @zhengpin | 是 @zhengpin |
| 学者领域图谱 | scholar_kg | 待定 | 是 @zhengpin | 是 @zhengpin |

## 2、待完成算子文件进度表 ❓
| 图谱类型 | 文件名 | 算子个数 | 负责同学 |
| --- | --- | --- | --- |
| sparql图谱 | sparql_kg | 待定 | @xuemeng |
| 超关系图谱 | hyper_relation_kg | 3 | @runhao |
| 医学领域图谱 | 待定 | 待定 | @wanpeng |

## 3、DataFlow-KG算子开发负责表 ✅
| 成员 | 已合并算子文件 |
| --- | --- |
| @runhao | tkg_4tuple_extractor.py, tkg_rel_4tuple_subgraph_qa_generator.py, tkg_4tuple_time_sampling.py, tkg_4tuple_time_summary.py, tkg_attri_4tuple_qa_generator.py, tkg_rel_4tuple_conversation_generator.py, tkg_attri_4tuple_qa_generator.py |
| @wanpeng | kg_rel_triple_consistency_eval.py, kg_rel_triple_topology_eval.py, kg_triple_hallucination_eval.py, mmkg_visual_triple_extractor.py, mmkg_visual_triple_path_qa_generator.py, mmkg_visual_triple_subgraph_qa_generator.py, mmkg_entity_link2database.py, mmkg_entity_link2img.py, mmkg_visual_triple_path_sampling.py, mmkg_visual_triple_subgraph_sampling.py |
| @xuemeng | 尽快合并 @zhengpin |
