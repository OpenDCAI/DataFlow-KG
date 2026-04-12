import pandas as pd
import pytest

from dataflow.core import LLMServingABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.operators.general_kg.eval.kg_qa_concise_eval import (
    KGQAConcisenessEvaluator,
)
from dataflow.operators.general_kg.filter.kg_entity_validation import (
    KGEntityValidity,
)


class FakeLLMServing(LLMServingABC):
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_from_input(self, user_inputs, system_prompt):
        self.calls.append(
            {
                "user_inputs": user_inputs,
                "system_prompt": system_prompt,
            }
        )
        return [self.responses.pop(0)]

    def start_serving(self):
        return None

    def cleanup(self):
        return None


class _TestStorage(DataFlowStorage):
    def __init__(self, data):
        self._data = data

    def get_keys_from_dataframe(self):
        return list(self._data.columns)

    def read(self, output_type="dataframe"):
        return self._data

    def write(self, data):
        self._data = data
        return data


@pytest.mark.cpu
def test_kgqa_conciseness_returns_custom_output_key():
    llm = FakeLLMServing(['{"conciseness_scores": [0.9, 0.2]}'])
    storage = _TestStorage(
        pd.DataFrame(
            {
                "pairs": [[
                    {"question": "Q1", "answer": "A1"},
                    {"question": "Q2", "answer": "A2"},
                ]]
            }
        )
    )

    operator = KGQAConcisenessEvaluator(llm_serving=llm, lang="en")
    output_keys = operator.run(
        storage=storage,
        input_key="pairs",
        output_key="scores",
    )

    result = storage.read("dataframe")

    assert output_keys == ["scores"]
    assert result["scores"].tolist() == [[0.9, 0.2]]


@pytest.mark.cpu
def test_kg_entity_validity_filters_entity_batches_to_new_column():
    llm = FakeLLMServing(['["Albert Einstein", "Paris"]'])
    storage = _TestStorage(
        pd.DataFrame(
            {
                "entity": ["Albert Einstein, Paris, xkqz123"],
            }
        )
    )

    operator = KGEntityValidity(llm_serving=llm, lang="en")
    output_keys = operator.run(
        storage=storage,
        input_key="entity",
        output_key="valid",
    )

    result = storage.read("dataframe")

    assert output_keys == ["valid"]
    assert result["valid"].tolist() == [["Albert Einstein", "Paris"]]
    assert "Albert Einstein, Paris, xkqz123" in llm.calls[0]["user_inputs"][0]


@pytest.mark.cpu
def test_kg_entity_validity_merge_to_input_overwrites_with_filtered_list():
    llm = FakeLLMServing(['```json\n["Albert Einstein", "Paris"]\n```'])
    storage = _TestStorage(
        pd.DataFrame(
            {
                "entity": ["Albert Einstein, Paris, xkqz123"],
            }
        )
    )

    operator = KGEntityValidity(
        llm_serving=llm,
        lang="en",
        merge_to_input=True,
    )
    output_keys = operator.run(
        storage=storage,
        input_key="entity",
        output_key="valid",
    )

    result = storage.read("dataframe")

    assert output_keys == ["entity"]
    assert result["entity"].tolist() == [["Albert Einstein", "Paris"]]
