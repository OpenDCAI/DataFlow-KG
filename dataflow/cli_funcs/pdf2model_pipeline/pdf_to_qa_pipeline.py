#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
from dataflow.operators.general_kg import (
    KGEntityExtraction,
    KGTripleExtraction,
    KGEntityBasedSubgraphSampling,
    KGRelationTripleSubgraphQAGeneration,
    QAExtractor
)
from dataflow.operators.pdf2text import (
    KBCTextCleaner,
    KBCChunkGenerator
)
from dataflow.utils.storage import FileStorage
from dataflow.serving import LocalModelLLMServing_vllm
from dataflow.serving import APILLMServing_request


class PDF2QA_GPUPipeline():
    def __init__(self, cache_base="./"):
        # 处理cache_base相对路径
        cache_path = Path(cache_base)
        if not cache_path.is_absolute():
            caller_cwd = Path(os.environ.get('PWD', os.getcwd()))
            cache_path = caller_cwd / cache_path

        self.storage = FileStorage(
            first_entry_file_name=str(cache_path / ".cache" / "gpu" / "wikipedia_3.json"),
            cache_path=str(cache_path / ".cache" / "gpu"),
            file_name_prefix="batch_cleaning_step",
            cache_type="json",
        )

        self.llm_serving = APILLMServing_request(
            api_url="http://123.129.219.111:3000/v1/chat/completions",
            model_name="gpt-4o-mini",
            max_workers=10,
        )

        self.qa_generation_step1 = KBCChunkGenerator(
            split_method="token",
            chunk_size=512,
            tokenizer_name="./Qwen2.5-7B-Instruct",
        )

        self.qa_generation_step2 = KBCTextCleaner(
            llm_serving=self.llm_serving,
            lang="en"
        )

        self.qa_generation_step3 = KGEntityExtraction(
            llm_serving=self.llm_serving,
            lang="en",
        )

        self.qa_generation_step4 = KGTripleExtraction(
            llm_serving=self.llm_serving,
            lang="en",
            triple_type="relation"
        )

        self.qa_generation_step5 = KGEntityBasedSubgraphSampling(
            llm_serving=self.llm_serving,
            lang="en",
        )

        self.qa_generation_step6 = KGRelationTripleSubgraphQAGeneration(
            llm_serving=self.llm_serving,
            lang="en",
            qa_type="num",
            num_q=5,
        )

        self.qa_generation_step7 = QAExtractor(
            output_json_file="./.cache/data/qa.json",
        )

    def forward(self):
        """执行完整的Pipeline流程"""
        print("🔄 Step 1: Text splitting into chunks...")
        self.qa_generation_step1.run(
            input_key="text",
            storage=self.storage.step(),
        )

        self.qa_generation_step2.run(
            storage=self.storage.step(),
            input_key="raw_chunk",
            output_key="cleaned_chunk",
        )

        print("🔄 Step 2: Knowledge graph construction...")
        self.qa_generation_step3.run(
            storage=self.storage.step(),
        )

        self.qa_generation_step4.run(
            storage=self.storage.step(),
            input_key="cleaned_chunk",
            output_key="triple"
        )

        print("🔄 Step 3: Subgraph sampling...")
        self.qa_generation_step5.run(
            storage=self.storage.step(),
            input_key="triple",
            output_key="subgraph",
            sampling_type="hop",
            hop=2,
            M=5,
        )

        print("🔄 Step 4: Multi-hop QA generation...")
        self.qa_generation_step6.run(
            storage=self.storage.step(),
            input_key="subgraph",
            output_key="QA_pairs",
        )

        self.qa_generation_step7.run(
            storage=self.storage.step(),
            input_qa_key="QA_pairs"
        )

        print("✅ Pipeline completed! Output saved to: ./.cache/data/qa.json")


def main():
    parser = argparse.ArgumentParser(description="PDF to QA Pipeline")
    parser.add_argument("--cache", default="./", help="Cache directory path")
    args = parser.parse_args()

    print("🚀 Starting PDF-to-QA Pipeline...")
    print(f"📄 Input: {args.cache}.cache/gpu/pdf_list.jsonl")
    print(f"💾 Cache: {args.cache}.cache/gpu/")
    print(f"📤 Output: {args.cache}.cache/data/qa.json")
    print("-" * 60)

    model = PDF2QA_GPUPipeline(cache_base=args.cache)
    model.forward()


if __name__ == "__main__":
    main()