import os

from dataflow.core import LLMServingABC
from dataflow.operators.general_kg.generate.kg_entity_extractor import (
    KGEntityExtraction,
)
from dataflow.operators.general_kg.generate.kg_triple_extractor import (
    KGTripleExtraction,
)
from dataflow.operators.multi_model_kg import (
    MMKGEntityBasedSubgraphSampling,
    MMKGSubgraphBaseQAGeneration,
    MMKGVisualTripleExtraction,
)
from dataflow.pipeline import PipelineABC
from dataflow.serving import APILLMServing_request, APIVLMServing_openai
from dataflow.utils.storage import FileStorage


class MultimodalKGPipeline(PipelineABC):
    """Multimodal KG pipeline: text triples + visual triples -> subgraphs -> QA pairs."""

    def __init__(
        self,
        first_entry_file_name: str,
        llm_serving: LLMServingABC,
        vlm_serving: LLMServingABC,
        cache_path: str = "./cache_local",
        file_name_prefix: str = "multimodal_kg_pipeline_step",
        cache_type: str = "json",
        lang: str = "en",
        triple_type: str = "relation",
        sampling_type: str = "hop",
        subgraph_hop: int = 2,
        quality_threshold: int = 3,
    ):
        super().__init__()
        if llm_serving is None:
            raise ValueError("llm_serving is required for MultimodalKGPipeline")
        if vlm_serving is None:
            raise ValueError("vlm_serving is required for MultimodalKGPipeline")

        self.storage = FileStorage(
            first_entry_file_name=first_entry_file_name,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )
        self.sampling_type = sampling_type
        self.subgraph_hop = subgraph_hop

        self.entity_extraction_step1 = KGEntityExtraction(
            llm_serving=llm_serving,
            lang=lang,
        )
        self.text_triple_extraction_step2 = KGTripleExtraction(
            llm_serving=llm_serving,
            triple_type=triple_type,
            lang=lang,
        )
        self.visual_triple_extraction_step3 = MMKGVisualTripleExtraction(
            llm_serving=vlm_serving,
            quality_threshold=quality_threshold,
            lang=lang,
        )
        self.subgraph_sampling_step4 = MMKGEntityBasedSubgraphSampling(
            llm_serving=llm_serving,
            lang=lang,
        )
        self.qa_generation_step5 = MMKGSubgraphBaseQAGeneration(
            llm_serving=vlm_serving,
            lang=lang,
        )

    def forward(self):
        self.entity_extraction_step1.run(
            storage=self.storage.step(),
            input_key="raw_chunk",
            output_key="entity",
        )
        self.text_triple_extraction_step2.run(
            storage=self.storage.step(),
            input_key="raw_chunk",
            input_key_meta="entity",
            output_key="triple",
        )
        self.visual_triple_extraction_step3.run(
            storage=self.storage.step(),
            input_key="img_dict",
            input_key_meta="entity",
            output_key="vis_triple",
        )
        self.subgraph_sampling_step4.run(
            storage=self.storage.step(),
            input_key="triple",
            output_key="subgraph",
            vis_triple_key="vis_triple",
            sampling_type=self.sampling_type,
            hop=self.subgraph_hop,
        )
        self.qa_generation_step5.run(
            storage=self.storage.step(),
            input_key="vis_url",
            input_key_meta="subgraph",
            output_key="QA_pairs",
        )


if __name__ == "__main__":
    input_file = os.environ.get("DF_MMKG_INPUT_FILE", "")
    if not input_file:
        raise ValueError(
            "Set DF_MMKG_INPUT_FILE to a JSON file containing `raw_chunk` and `img_dict`."
        )

    llm_serving = APILLMServing_request(
        api_url="https://api.openai.com/v1/chat/completions",
        key_name_of_api_key="DF_API_KEY",
        model_name="gpt-4o-mini",
        max_workers=6,
        temperature=0.0,
    )
    vlm_serving = APIVLMServing_openai(
        api_url="https://api.openai.com/v1",
        key_name_of_api_key="DF_API_KEY",
        model_name="o4-mini",
        max_workers=4,
        temperature=0.0,
    )

    pipeline = MultimodalKGPipeline(
        first_entry_file_name=input_file,
        llm_serving=llm_serving,
        vlm_serving=vlm_serving,
        cache_path="./cache_mmkg",
        lang="en",
        sampling_type="hop",
        subgraph_hop=2,
    )
    pipeline.forward()
