import os
from dataflow.operators.general_kg.eval.kg_rel_triple_topology_eval import KGRelationTripleTopologyEvaluator
from dataflow.operators.general_kg.eval.kg_subgraph_scale_eval import KGSubgraphScaleEvaluator
from dataflow.operators.general_kg.eval.kg_subgraph_connectivity_eval import KGSubgraphConnectivityEvaluator
from dataflow.operators.general_kg.eval.kg_rel_triple_consistency_eval import KGRelationTripleConsistencyEvaluator
from dataflow.operators.general_kg.eval.kg_rel_triple_strength_eval import KGRelationStrengthScoring
from dataflow.operators.general_kg.eval.kg_rel_triple_nx_visual import KGRelationTripleVisualization
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request


class KGEvaluationVisualizationPipeline:
    """通用知识图谱评测与可视化流水线"""

    def __init__(self, lang: str = "en"):
        # -------- Storage --------
        self.storage = FileStorage(
            first_entry_file_name="../example_data/KGEvaluationPipeline/input.json",
            cache_path="./kg_evaluation",
            file_name_prefix="kg_eval",
            cache_type="json",
        )

        # -------- LLM Serving --------
        self.llm_serving = APILLMServing_request(
            api_url="http://123.129.219.111:3000/v1/chat/completions",
            model_name="gpt-4o",
            max_workers=20,
        )

        # -------- 无需 LLM 的评测算子 --------
        self.topology_eval = KGRelationTripleTopologyEvaluator()
        self.scale_eval = KGSubgraphScaleEvaluator()
        self.connectivity_eval = KGSubgraphConnectivityEvaluator()

        # -------- 需要 LLM 的评测算子 --------
        self.consistency_eval = KGRelationTripleConsistencyEvaluator(
            llm_serving=self.llm_serving,
            sample_rate=1.0,
            max_samples=10,
            lang=lang,
        )
        self.strength_eval = KGRelationStrengthScoring(
            llm_serving=self.llm_serving,
            lang=lang,
        )

        # -------- 可视化算子 --------
        self.visualization = KGRelationTripleVisualization(lang=lang)

    def forward(self):
        """依次执行所有评测算子和可视化算子"""

        print("=" * 60)
        print("Step 1/6: 拓扑结构评测 (Topology Evaluation)")
        print("=" * 60)
        self.topology_eval.run(
            storage=self.storage.step(),
            input_key="triple",
        )

        print("\n" + "=" * 60)
        print("Step 2/6: 子图规模评测 (Subgraph Scale Evaluation)")
        print("=" * 60)
        self.scale_eval.run(
            storage=self.storage.step(),
            input_key="triple",
        )

        print("\n" + "=" * 60)
        print("Step 3/6: 子图连通性评测 (Subgraph Connectivity Evaluation)")
        print("=" * 60)
        self.connectivity_eval.run(
            storage=self.storage.step(),
            input_key="triple",
        )

        print("\n" + "=" * 60)
        print("Step 4/6: 三元组逻辑一致性评测 (Consistency Evaluation) [LLM]")
        print("=" * 60)
        self.consistency_eval.run(
            storage=self.storage.step(),
            input_key="triple",
        )

        print("\n" + "=" * 60)
        print("Step 5/6: 三元组语义强度评分 (Strength Scoring) [LLM]")
        print("=" * 60)
        self.strength_eval.run(
            storage=self.storage.step(),
            input_key="raw_chunk",
            input_key_meta="triple",
            output_key="triple_strength_score",
        )

        print("\n" + "=" * 60)
        print("Step 6/6: 知识图谱可视化 (KG Visualization)")
        print("=" * 60)
        visual_html = os.path.join(self.storage.cache_path, "kg_visualization.html")
        self.visualization.run(
            storage=self.storage.step(),
            input_key="triple",
            output_key="kg_visualization",
            output_html=visual_html,
        )

        # -------- 打印结果摘要 --------
        self._print_summary()

    def _print_summary(self):
        """读取最终结果并打印评测摘要"""
        step6_file = os.path.join(self.storage.cache_path, "kg_eval_step6.json")
        try:
            if os.path.exists(step6_file):
                df = self.storage.read("dataframe", file_path=step6_file)
            else:
                df = self.storage.read("dataframe")
        except Exception as e:
            print(f"\n[Warning] 无法读取最终结果: {e}")
            return

        print("\n" + "=" * 60)
        print("评测结果摘要 (Evaluation Summary)")
        print("=" * 60)

        # 拓扑指标
        topo_keys = ["lcc_ratio", "structure_avg_degree", "fragmentation_score",
                     "num_components", "node_count", "edge_count"]
        print("\n[拓扑结构指标]")
        for key in topo_keys:
            if key in df.columns:
                print(f"  {key}: {df[key].tolist()}")

        # 规模指标
        scale_keys = ["num_nodes", "num_edges", "density"]
        print("\n[子图规模指标]")
        for key in scale_keys:
            if key in df.columns:
                print(f"  {key}: {df[key].tolist()}")

        # 连通性指标
        conn_keys = ["vertex_connectivity", "edge_connectivity", "global_efficiency"]
        print("\n[子图连通性指标]")
        for key in conn_keys:
            if key in df.columns:
                print(f"  {key}: {df[key].tolist()}")

        # LLM 评测指标
        print("\n[LLM 评测指标]")
        if "logical_consistency_score" in df.columns:
            print(f"  logical_consistency_score: {df['logical_consistency_score'].tolist()}")
        if "triple_strength_score" in df.columns:
            print(f"  triple_strength_score: {df['triple_strength_score'].tolist()}")
        if "kg_visualization" in df.columns:
            print(f"  kg_visualization: {df['kg_visualization'].iloc[0]}")

        print("\n" + "=" * 60)
        print(f"缓存文件目录: {self.storage.cache_path}")
        print("=" * 60)


if __name__ == "__main__":
    pipeline = KGEvaluationVisualizationPipeline()
    pipeline.forward()
