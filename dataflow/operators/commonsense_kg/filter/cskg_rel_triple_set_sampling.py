import pandas as pd
import random
import re
import Levenshtein
import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque
from fuzzywuzzy import fuzz
from multiprocessing import Pool, cpu_count
import itertools

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC


@OPERATOR_REGISTRY.register()
class CSKGRelationTripleSetSampling(OperatorABC):
    """
    High-efficiency Knowledge Graph related triple sets generation operator.
    Optimized for LARGE-SCALE triples (100k+), supports:
    1. Pre-indexing (hash-based) for fast matching
    2. Batch similarity calculation
    3. Parallel processing
    4. Memory-efficient data structures
    5. Filter out sets with only one triple
    """
    def __init__(
        self,
        llm_serving: LLMServingABC,
        seed: int = 0,
        lang: str = "en",
        num_q: int = 5,
        n_jobs: int = -1  # 并行进程数（-1=使用所有CPU核心）
    ):
        self.rng = random.Random(seed)
        self.lang = lang
        self.num_q = num_q
        self.logger = get_logger()
        self.n_jobs = cpu_count() if n_jobs == -1 else max(1, n_jobs)

        # 三元组解析正则
        self.triple_pattern = re.compile(
            r"<subj>\s*(.+?)\s*<obj>\s*(.+?)\s*<rel>\s*(.+?)$"
        )

        # 预定义批量相似度计算的分块大小（平衡内存和速度）
        self.batch_size = 1000


    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        """
        Return a short description of the operator.
        """
        if lang == "zh":
            return (
                "CSKGRelationTripleSetSampling 是一个高效的三元组集合采样算子，"
                "用于将大规模离散三元组按匹配规则（如相似主体、相似客体或相同关系）聚合成相关的三元组集合，并自动过滤单元素集合。",
                "输入为离散的三元组（triple），输出为聚合后的相关三元组集合列表（set_triple）。"
            )
        else:
            return (
                "CSKGRelationTripleSetSampling is a high-efficiency operator that groups large-scale discrete triples into related sets "
                "based on matching rules (e.g., similar subjects/objects or identical relations), automatically filtering out single-element sets.",
                "Input: discrete triples. Output: aggregated related triple sets (set_triple)."
            )
        
    def _parse_triple(self, triple_str: str) -> tuple[str, str, str]:
        """轻量解析（减少内存占用）"""
        match = self.triple_pattern.match(triple_str.strip())
        if not match:
            return "", "", ""  # 跳过无效三元组（避免中断）
        head, tail, relation = match.groups()
        return head.strip(), relation.strip(), tail.strip()

    def _batch_calculate_similarity(self, str_list: List[str], target_str: str) -> np.ndarray:
        """
        批量计算相似度（向量化操作，替代逐一遍历）
        :param str_list: 待批量计算的字符串列表
        :param target_str: 目标字符串
        :return: 相似度数组（np.ndarray, 0-1）
        """
        # 归一化目标字符串
        target_norm = target_str.lower().replace(" ", "") if self.lang == "en" else target_str.replace(" ", "")
        
        # 批量归一化待比较字符串
        str_list_norm = []
        for s in str_list:
            s_norm = s.lower().replace(" ", "") if self.lang == "en" else s.replace(" ", "")
            str_list_norm.append(s_norm)
        
        # 批量计算编辑距离（向量化）
        lev_sims = np.array([Levenshtein.ratio(s, target_norm) for s in str_list_norm])
        
        # 批量计算模糊匹配度
        fuzzy_sims = np.array([fuzz.partial_ratio(s, target_str) / 100.0 for s in str_list])
        
        # 加权融合
        final_sims = (lev_sims * 0.7) + (fuzzy_sims * 0.3)
        return final_sims



    def _build_indexes(self, parsed_triples: List[Tuple[str, str, str, str]]) -> Dict[str, Any]:
        """
        预构建哈希索引，支持快速匹配：
        - rel2triples: rel -> 三元组列表（规则3的O(1)查找）
        - subj2triples: subj -> 三元组列表（规则1的候选缩小）
        - obj2triples: obj -> 三元组列表（规则2的候选缩小）
        - str2idx: 字符串 -> 索引（去重+快速映射）
        """
        indexes = {
            "rel2triples": defaultdict(list),
            "subj2triples": defaultdict(list),
            "obj2triples": defaultdict(list),
            "str2idx": {},
            "idx2triple": [],
            "all_subjs": [],
            "all_objs": [],
            "all_subj_idx": defaultdict(list),
            "all_obj_idx": defaultdict(list)
        }

        # 构建索引（单次遍历，O(n)）
        for idx, (subj, rel, obj, t_str) in enumerate(parsed_triples):
            # 去重（避免重复处理）
            if t_str in indexes["str2idx"]:
                continue
            indexes["str2idx"][t_str] = idx
            indexes["idx2triple"].append(t_str)
            
            # 关系索引（规则3核心）
            indexes["rel2triples"][rel].append(t_str)
            
            # 主体/客体索引（规则1/2的候选缩小）
            indexes["subj2triples"][subj].append(t_str)
            indexes["obj2triples"][obj].append(t_str)
            
            # 主体/客体的字符串+索引映射（批量相似度计算用）
            indexes["all_subjs"].append(subj)
            indexes["all_objs"].append(obj)
            indexes["all_subj_idx"][subj].append(idx)
            indexes["all_obj_idx"][obj].append(idx)

        # 转换为numpy数组（加速批量计算）
        indexes["all_subjs_arr"] = np.array(indexes["all_subjs"])
        indexes["all_objs_arr"] = np.array(indexes["all_objs"])

        self.logger.info(f"Index built | total unique triples={len(indexes['idx2triple'])} | "
                         f"unique rels={len(indexes['rel2triples'])} | "
                         f"unique subjs={len(indexes['subj2triples'])}")
        return indexes


    def _process_chunk(self, args: Tuple[List[str], Dict[str, Any], int, float]) -> List[List[str]]:
        """
        单分片处理函数（供多进程调用）
        :param args: (chunk_triples, indexes, match_rule, similarity_threshold)
        :return: 分片内的相关集合列表（已过滤单元素集合）
        """
        chunk_triples, indexes, match_rule, similarity_threshold = args
        chunk_related_sets = []
        seen_set_ids = set()

        for ref_triple in chunk_triples:
            if not ref_triple:
                continue
            
            # 解析参考三元组
            ref_subj, ref_rel, ref_obj = self._parse_triple(ref_triple)
            if not (ref_subj and ref_rel and ref_obj):
                continue

            # 按规则快速生成相关集合
            if match_rule == 3:
                # 规则3：相同关系（O(1) 查找，无需遍历）
                related_triples = indexes["rel2triples"].get(ref_rel, [])
            
            else:
                # 规则1/2：相似主体/客体（先缩小候选+批量计算）
                if match_rule == 1:
                    # 步骤1：缩小候选范围（同前缀/同长度的subj）
                    candidate_subjs = [s for s in indexes["subj2triples"].keys() 
                                      if len(s) > 0 and abs(len(s) - len(ref_subj)) <= 2]
                    if not candidate_subjs:
                        related_triples = []
                        continue
                    
                    # 步骤2：批量计算相似度
                    sims = self._batch_calculate_similarity(candidate_subjs, ref_subj)
                    # 步骤3：筛选相似度>阈值的主体，合并对应的三元组
                    high_sim_subjs = [candidate_subjs[i] for i in np.where(sims > similarity_threshold)[0]]
                    related_triples = list(itertools.chain(*[indexes["subj2triples"][s] for s in high_sim_subjs]))
                
                else:  # match_rule == 2
                    # 规则2：相似客体（同规则1逻辑）
                    candidate_objs = [o for o in indexes["obj2triples"].keys() 
                                     if len(o) > 0 and abs(len(o) - len(ref_obj)) <= 2]
                    if not candidate_objs:
                        related_triples = []
                        continue
                    
                    sims = self._batch_calculate_similarity(candidate_objs, ref_obj)
                    high_sim_objs = [candidate_objs[i] for i in np.where(sims > similarity_threshold)[0]]
                    related_triples = list(itertools.chain(*[indexes["obj2triples"][o] for o in high_sim_objs]))

            # 第一步过滤：移除仅含单个三元组的集合
            if len(related_triples) <= 1:
                continue
            
            # 去重+排序（用于集合去重）
            related_triples = sorted(list(set(related_triples)))
            set_id = "||".join(related_triples)
            
            # 避免重复集合
            if set_id not in seen_set_ids:
                chunk_related_sets.append(related_triples)
                seen_set_ids.add(set_id)

        return chunk_related_sets


    def _generate_all_related_triple_sets(
        self,
        triple_groups: List[List[str]],
        match_rule: int,
        similarity_threshold: float = 0.7,
        deduplicate_sets: bool = True,
        chunk_size: int = 5000  # 分片大小（控制内存占用）
    ) -> List[List[str]]:
        """
        高效生成全量相关集合（适配大规模三元组）
        步骤：1. 解析+去重 2. 预构建索引 3. 分片+并行处理 4. 合并结果+二次过滤
        """
        # 步骤1：解析并去重所有三元组（单次遍历）
        self.logger.info("Parsing and deduplicating triples...")
        parsed_triples = []
        seen_triples = set()

        for group in triple_groups:
            for t_str in group:
                if t_str in seen_triples:
                    continue
                seen_triples.add(t_str)
                subj, rel, obj = self._parse_triple(t_str)
                if subj and rel and obj:  # 过滤无效三元组
                    parsed_triples.append((subj, rel, obj, t_str))

        total_triples = len(parsed_triples)
        if total_triples == 0:
            self.logger.warning("No valid triples found")
            return []
        self.logger.info(f"Total unique valid triples: {total_triples}")

        # 步骤2：预构建索引（O(n) 时间）
        self.logger.info("Building hash indexes for fast matching...")
        indexes = self._build_indexes(parsed_triples)

        # 步骤3：拆分参考三元组为分片（控制内存+支持并行）
        self.logger.info(f"Splitting triples into chunks (size={chunk_size}) for parallel processing...")
        ref_triples = [t[3] for t in parsed_triples]  # 参考三元组列表
        chunks = [ref_triples[i:i+chunk_size] for i in range(0, len(ref_triples), chunk_size)]
        self.logger.info(f"Split into {len(chunks)} chunks | using {self.n_jobs} processes")

        # 步骤4：并行处理所有分片
        self.logger.info("Processing chunks in parallel...")
        pool = Pool(processes=self.n_jobs)
        # 构造每个分片的参数
        chunk_args = [
            (chunk, indexes, match_rule, similarity_threshold)
            for chunk in chunks
        ]
        # 并行执行
        chunk_results = pool.map(self._process_chunk, chunk_args)
        pool.close()
        pool.join()

        # 步骤5：合并分片结果并全局去重 + 二次过滤单元素集合
        self.logger.info("Merging chunk results...")
        all_related_sets = []
        global_seen_set_ids = set()

        for chunk_res in chunk_results:
            for related_set in chunk_res:
                # 二次过滤：防止分片内漏过滤的单元素集合
                if len(related_set) <= 1:
                    continue
                    
                if not deduplicate_sets:
                    all_related_sets.append(related_set)
                    continue
                # 全局去重
                set_id = "||".join(sorted(related_set))
                if set_id not in global_seen_set_ids:
                    all_related_sets.append(related_set)
                    global_seen_set_ids.add(set_id)

        self.logger.info(f"Final result | total unique sets (filtered): {len(all_related_sets)}")
        return all_related_sets


    def run(
        self,
        storage: DataFlowStorage = None,
        input_key: str = "triple",
        output_key: str = "set_triple",
        match_rule: int = 1,
        similarity_threshold: float = 0.7,
        deduplicate_sets: bool = True,
        chunk_size: int = 5000  # 新增：分片大小（大规模场景建议5000-10000）
    ):
        """
        运行算子（适配大规模三元组）
        :param chunk_size: 分片大小（越小内存占用越低，建议根据机器配置调整）
        """
        self.input_key = input_key
        self.output_key = output_key

        # 读取数据
        dataframe = storage.read("dataframe")
        if self.input_key not in dataframe.columns:
            raise ValueError(f"Missing required column: {self.input_key}")
        if self.output_key in dataframe.columns:
            raise ValueError(f"Column already exists: {self.output_key}")

        # 提取三元组
        triple_groups: List[List[str]] = dataframe[self.input_key].tolist()

        # 高效生成全量相关集合
        all_related_sets = self._generate_all_related_triple_sets(
            triple_groups=triple_groups,
            match_rule=match_rule,
            similarity_threshold=similarity_threshold,
            deduplicate_sets=deduplicate_sets,
            chunk_size=chunk_size
        )

        # 写入结果（内存高效的方式）
        df = pd.DataFrame({self.output_key: [all_related_sets]})
        output_file = storage.write(df)
        
        self.logger.info(
            f"Large-scale related triple sets generation done | "
            f"rule={match_rule} | threshold={similarity_threshold} | "
            f"total filtered sets={len(all_related_sets)} | saved to {output_file}"
        )

        return [output_key]