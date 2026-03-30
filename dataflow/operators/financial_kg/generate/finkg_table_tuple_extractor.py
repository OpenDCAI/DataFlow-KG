# -*- coding: utf-8 -*-
"""
====================================
DataFlow-KG: FinKGTableTupleExtraction
====================================

License:
    MIT License
"""

import json
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from dataflow import get_logger
from dataflow.core import LLMServingABC, OperatorABC
from dataflow.core.prompt import prompt_restrict
from dataflow.prompts.diverse_kg.finkg import (
    FinKGTableSchemaPrompt,
    FinKGTableTupleExtractionPrompt,
)
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage, FileStorage


class FinKGTableTupleExtractionLLM:
    r"""
    Handle table normalization, schema inference, and tuple extraction for
    arbitrary financial tables.
    """

    def __init__(
        self,
        llm_serving: Optional[LLMServingABC],
        lang: str = "en",
        max_table_chars: int = 12000,
    ):
        self.llm_serving = llm_serving
        self.lang = lang
        self.max_table_chars = max_table_chars
        self.logger = get_logger()
        self.schema_prompt = FinKGTableSchemaPrompt(lang=lang)
        self.tuple_prompt = FinKGTableTupleExtractionPrompt(lang=lang)

    def llm_query(
        self,
        raw_data: List[Dict[str, Any]],
        ontology: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        r"""Extract tuples from arbitrary financial table chunks."""

        if self.llm_serving is None:
            raise ValueError("llm_serving is required for table-to-KG extraction")

        self.logger.info("Starting financial table-to-KG extraction...")
        results = []

        for data in tqdm(raw_data, desc="Extract tuples from tables"):
            table_text = self._normalize_table_input(data.get("table"))
            table_text = self._truncate_table_text(table_text)
            table_title = self._normalize_optional_text(data.get("table_title"))
            table_context = self._normalize_optional_text(data.get("table_context"))

            if not table_text:
                results.append(
                    {
                        "table_schema": self._empty_schema(),
                        "tuple": [],
                        "entity_class": [],
                    }
                )
                continue

            schema = self._infer_schema(
                table_text=table_text,
                table_title=table_title,
                table_context=table_context,
                ontology=ontology,
            )

            extracted = self._extract_tuples(
                table_text=table_text,
                table_title=table_title,
                table_context=table_context,
                schema=schema,
                ontology=ontology,
            )

            results.append(
                {
                    "table_schema": schema,
                    "tuple": extracted["tuple"],
                    "entity_class": extracted["entity_class"],
                }
            )

        return results

    def _infer_schema(
        self,
        table_text: str,
        table_title: str,
        table_context: str,
        ontology: Dict[str, Any],
    ) -> Dict[str, Any]:
        user_prompt = self.schema_prompt.build_prompt(
            table_text=table_text,
            table_title=table_title,
            table_context=table_context,
        )
        system_prompt = self.schema_prompt.build_system_prompt(ontology)

        responses = self.llm_serving.generate_from_input(
            user_inputs=[user_prompt],
            system_prompt=system_prompt,
        )

        return self._parse_schema_response(responses[0] if responses else "")

    def _extract_tuples(
        self,
        table_text: str,
        table_title: str,
        table_context: str,
        schema: Dict[str, Any],
        ontology: Dict[str, Any],
    ) -> Dict[str, List[Any]]:
        schema_json = json.dumps(schema, ensure_ascii=False, indent=2)
        user_prompt = self.tuple_prompt.build_prompt(
            table_text=table_text,
            schema_json=schema_json,
            table_title=table_title,
            table_context=table_context,
        )
        system_prompt = self.tuple_prompt.build_system_prompt(ontology)

        responses = self.llm_serving.generate_from_input(
            user_inputs=[user_prompt],
            system_prompt=system_prompt,
        )

        return self._parse_tuple_response(responses[0] if responses else "")

    def _normalize_table_input(self, table_input: Any) -> str:
        if table_input is None:
            return ""

        if isinstance(table_input, str):
            return table_input.strip()

        if isinstance(table_input, dict):
            return json.dumps(table_input, ensure_ascii=False, indent=2)

        if isinstance(table_input, list):
            return json.dumps(table_input, ensure_ascii=False, indent=2)

        return str(table_input).strip()

    def _normalize_optional_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and pd.isna(value):
            return ""
        return str(value).strip()

    def _truncate_table_text(self, table_text: str) -> str:
        if len(table_text) <= self.max_table_chars:
            return table_text

        truncated = table_text[: self.max_table_chars].rstrip()
        return f"{truncated}\n... [TRUNCATED]"

    def _strip_code_fence(self, response: str) -> str:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.replace("```json", "", 1)
            cleaned = cleaned.replace("```", "")
        return cleaned.strip()

    def _parse_schema_response(self, response: str) -> Dict[str, Any]:
        schema = self._empty_schema()
        try:
            data = json.loads(self._strip_code_fence(response))
        except Exception as exc:
            self.logger.warning(f"Failed to parse table schema response: {exc}")
            return schema

        for key in schema:
            value = data.get(key, schema[key])
            schema[key] = value

        if not isinstance(schema["candidate_entity_types"], dict):
            schema["candidate_entity_types"] = {}

        for key in [
            "primary_entity_columns",
            "secondary_entity_columns",
            "time_columns",
            "relation_columns",
            "attribute_columns",
            "value_columns",
            "candidate_relations",
            "candidate_attributes",
        ]:
            if not isinstance(schema[key], list):
                schema[key] = []

        if not isinstance(schema["table_type"], str):
            schema["table_type"] = "unknown"

        if not isinstance(schema["row_semantics"], str):
            schema["row_semantics"] = ""

        return schema

    def _parse_tuple_response(self, response: str) -> Dict[str, List[Any]]:
        empty = {"tuple": [], "entity_class": []}

        try:
            data = json.loads(self._strip_code_fence(response))
        except Exception as exc:
            self.logger.warning(f"Failed to parse table tuple response: {exc}")
            return empty

        raw_tuples = data.get("tuple", [])
        raw_classes = data.get("entity_class", [])

        if not isinstance(raw_tuples, list) or not isinstance(raw_classes, list):
            return empty

        tuples: List[str] = []
        classes: List[List[str]] = []

        for tuple_item, class_item in zip(raw_tuples, raw_classes):
            if not isinstance(tuple_item, str):
                continue
            if not isinstance(class_item, list):
                continue
            if not all(isinstance(cls, str) for cls in class_item):
                continue

            tuples.append(tuple_item.strip())
            classes.append([cls.strip() for cls in class_item])

        return {
            "tuple": tuples,
            "entity_class": classes,
        }

    def _empty_schema(self) -> Dict[str, Any]:
        return {
            "table_type": "unknown",
            "primary_entity_columns": [],
            "secondary_entity_columns": [],
            "time_columns": [],
            "relation_columns": [],
            "attribute_columns": [],
            "value_columns": [],
            "candidate_entity_types": {},
            "candidate_relations": [],
            "candidate_attributes": [],
            "row_semantics": "",
        }


@prompt_restrict(
    FinKGTableSchemaPrompt,
    FinKGTableTupleExtractionPrompt,
)
@OPERATOR_REGISTRY.register()
class FinKGTableTupleExtraction(OperatorABC):
    r"""
    Convert arbitrary financial tables into Financial KG tuples using LLM.

    Input columns:
      - raw_table: serialized table content
      - table_title: optional table title
      - table_context: optional table context

    Output columns:
      - tuple
      - entity_class
      - table_schema
    """

    def __init__(
        self,
        llm_serving: LLMServingABC,
        lang: str = "en",
        max_table_chars: int = 12000,
    ):
        self.logger = get_logger()
        self.lang = lang
        self.llm_serving = llm_serving
        self.extractor = FinKGTableTupleExtractionLLM(
            llm_serving=llm_serving,
            lang=lang,
            max_table_chars=max_table_chars,
        )

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "FinKGTableTupleExtraction 用于将任意金融表格转换为金融知识图谱四元组。",
                "输入: raw_table + table_title + table_context + ontology; 输出: tuple + entity_class + table_schema",
            )
        return (
            "FinKGTableTupleExtraction is used to convert arbitrary financial tables into Financial KG tuples.",
            "Input: raw_table + table_title + table_context + ontology; Output: tuple + entity_class + table_schema",
        )

    def process_batch(
        self,
        raw_tables: List[Any],
        ontology: Dict[str, Any],
        table_titles: Optional[List[Any]] = None,
        table_contexts: Optional[List[Any]] = None,
        sources: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        num_rows = len(raw_tables)

        if table_titles is None:
            table_titles = [""] * num_rows
        if table_contexts is None:
            table_contexts = [""] * num_rows
        if sources is None:
            sources = ["default_source"] * num_rows

        raw_data = [
            {
                "table": table,
                "table_title": title,
                "table_context": context,
                "source": source,
            }
            for table, title, context, source in zip(
                raw_tables, table_titles, table_contexts, sources
            )
        ]

        return self.extractor.llm_query(raw_data=raw_data, ontology=ontology)

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        if self.input_key not in dataframe.columns:
            raise ValueError(f"Missing required column: {self.input_key}")

        for column in [self.output_key, self.output_key_meta, self.output_schema_key]:
            if column in dataframe.columns:
                raise ValueError(f"Output column already exists: {column}")

    def run(
        self,
        storage: DataFlowStorage = None,
        ontology_lists: Optional[Dict[str, Any]] = None,
        input_key: str = "raw_table",
        input_title_key: Optional[str] = "table_title",
        input_context_key: Optional[str] = "table_context",
        input_key_meta: str = "finkg_ontology",
        output_key: str = "tuple",
        output_key_meta: str = "entity_class",
        output_schema_key: str = "table_schema",
    ) -> List[str]:
        self.input_key = input_key
        self.output_key = output_key
        self.output_key_meta = output_key_meta
        self.output_schema_key = output_schema_key

        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        raw_tables = dataframe[self.input_key].tolist()

        table_titles = None
        if input_title_key and input_title_key in dataframe.columns:
            table_titles = dataframe[input_title_key].tolist()

        table_contexts = None
        if input_context_key and input_context_key in dataframe.columns:
            table_contexts = dataframe[input_context_key].tolist()

        if ontology_lists is None:
            storage_meta = FileStorage(
                first_entry_file_name="",
                cache_type="json",
            )
            ontology_df = storage_meta.read(
                file_path=f"./.cache/api/{input_key_meta}.json",
                output_type="dataframe",
            )
            row = ontology_df.iloc[0]
            ontology = {
                "entity_type": row["entity_type"],
                "relation_type": row["relation_type"],
                "attribute_type": row.get("attribute_type", {}),
            }
        else:
            ontology = ontology_lists

        outputs = self.process_batch(
            raw_tables=raw_tables,
            ontology=ontology,
            table_titles=table_titles,
            table_contexts=table_contexts,
        )

        dataframe[self.output_schema_key] = [
            output.get("table_schema", {})
            for output in outputs
        ]
        dataframe[self.output_key] = [
            output.get("tuple", [])
            for output in outputs
        ]
        dataframe[self.output_key_meta] = [
            output.get("entity_class", [])
            for output in outputs
        ]

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")

        return [self.output_key, self.output_key_meta, self.output_schema_key]
