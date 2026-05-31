import os
import time
import uuid
import openai

# from dotenv import load_dotenv, find_dotenv
from tenacity import (
    retry,
    wait_random_exponential,
    before_sleep_log,
    stop_after_attempt,
)
import logging
import sys
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Union, Optional
import tenacity
import httpx

# Configure logging
logger = logging.getLogger("OpenAIUtils")
logger.setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

# _ = load_dotenv(find_dotenv())
# OpenAI
MAX_ATTEMPTS = 1


def _response_to_serializable(response: Any) -> Any:
    if response is None:
        return None
    if hasattr(response, "model_dump"):
        try:
            return response.model_dump()
        except Exception:
            pass
    if isinstance(response, (dict, list, str, int, float, bool)):
        return response
    return {"_fallback_repr": repr(response)}


def _write_wikontic_api_json_bundle(
    log_dir: str,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    response: Any,
    usage: Any,
    assistant_content_raw: str,
    extra_request_fields: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """将单次调用的 request + 完整 response 等写入一个 JSON 文件。"""
    d = Path(log_dir)
    d.mkdir(parents=True, exist_ok=True)
    fname = f"api_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:12]}.json"
    out = d / fname
    req: Dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "messages": messages,
    }
    if extra_request_fields:
        req.update(extra_request_fields)
    bundle: Dict[str, Any] = {
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "request": req,
        "response": _response_to_serializable(response),
        "usage": _response_to_serializable(usage),
        "assistant_content_raw": assistant_content_raw,
    }
    with out.open("w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2, default=str)
    return out


def _log_long(prefix: str, payload: Union[str, dict, list, None], chunk_size: int = 12000) -> None:
    """分块写入长文本/对象 JSON，避免单行过长。"""
    if payload is None:
        s = ""
    elif isinstance(payload, str):
        s = payload
    elif isinstance(payload, (dict, list)):
        s = json.dumps(payload, ensure_ascii=False, default=str)
    else:
        s = repr(payload)
    cs = max(800, int(chunk_size))
    logger.info("%s chars=%d", prefix, len(s))
    for i in range(0, len(s), cs):
        logger.info("%s", s[i : i + cs])


class LLMTripletExtractor:
    """A class for extracting and processing knowledge graph triplets using OpenAI's LLMs."""

    MODEL_PRICES = {
        "gpt-4o": {"input": 2.5, "output": 10},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
        "gpt-4.1-mini": {"input": 0.4, "output": 1.6},
        "gpt-4.1": {"input": 2.0, "output": 8.0},
        "Meta-llama/Llama-3.3-70B-Instruct": {"input": 0.04, "output": 0.12},
        "qwen/qwen3-32b": {"input": 0.05, "output": 0.2},
        "Openai/Gpt-oss-120b": {"input": 0.05, "output": 0.2},
        "Qwen/Qwen3-32B": {"input": 0.05, "output": 0.2},
        # 本地/网关别名（config.md：modelname 写 qwen32b、llama3.1-70b-awq）；单价与同类开源模型对齐，仅用于日志估算
        "qwen32b": {"input": 0.05, "output": 0.2},
        "llama3.1-70b-awq": {"input": 0.04, "output": 0.12},
        "gpt-3.5-turbo": {"input": 0.0, "output": 0.0},
        "deepseek-v3": {"input": 1.33, "output": 5.33},
        "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    }

    def __init__(
        self,
        api_key: str,
        prompt_folder_path: str = str(Path(__file__).parent / "prompts"),
        system_prompt_paths: Optional[Dict[str, str]] = None,
        model: str = "gpt-4o",
        max_attempts=MAX_ATTEMPTS,
        proxy: str = None,
        base_url: str = "https://api.openai.com/v1",
        gemini_reasoning_effort: Optional[str] = None,
    ):
        if proxy:
            http_client = httpx.Client(proxy=proxy)
            self.client = openai.OpenAI(
                api_key=api_key, http_client=http_client, base_url=base_url
            )
        else:
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

        """
        Initialize the LLMTripletExtractor.

        Args:
            prompt_folder_path: Path to folder containing prompt files
            system_prompt_paths: Dictionary mapping prompt types to file paths
            model: Name of the OpenAI model to use
        """
        if system_prompt_paths is None:
            system_prompt_paths = {
                "triplet_extraction": "triplet_extraction/propmt_1_types_qualifiers.txt",
                # 'triplet_extraction': 'triplet_extraction/prompt_1_types_qualifiers_dialog_bench.txt',
                "relation_entity_types_ranker": "ontology_refinement/prompt_choose_relation_and_types.txt",
                "relation_ranker": "ontology_refinement/prompt_choose_relation.txt",
                "entity_types_ranker": "ontology_refinement/prompt_choose_entity_types.txt",
                "relation_ranker_wo_entity_types": "name_refinement/prompt_choose_relation_wo_entity_types.txt",
                # 'relation_ranker_wo_entity_types': 'name_refinement/prompt_choose_relation_wo_entity_types_dialog_bench.txt',
                # 'subject_ranker': 'name_refinement/rank_subject_names_dialog_bench.txt',
                "subject_ranker": "name_refinement/rank_subject_names.txt",
                # 'object_ranker': 'name_refinement/rank_object_names_dialog_bench.txt',
                "object_ranker": "name_refinement/rank_object_names.txt",
                "quailfier_object_ranker": "name_refinement/rank_object_qualifiers.txt",
                "question_entity_extractor": "qa/prompt_entity_extraction_from_question.txt",
                "question_entity_ranker": "qa/prompt_choose_relevant_entities_for_question.txt",
                "question_entity_ranker_wo_types": "qa/prompt_choose_relevant_entities_for_question_wo_types.txt",
                # 'qa': 'qa_prompt_hotpot.txt'
                "question_decomposition_1": "qa/question_decomposition_1.txt",
                "qa_collapsing": "qa/qa_collapsing_prompt.txt",
                "qa_is_answered": "qa/prompt_is_answered.txt",
                "qa": "qa/qa_prompt.txt",
            }

        # Load all prompts
        prompt_folder = Path(prompt_folder_path)
        self.prompts = {}
        for prompt_type, filename in system_prompt_paths.items():
            with open(prompt_folder / filename) as f:
                self.prompts[prompt_type] = f.read()

        self.model = model
        # None：未从调用方指定，get_completion 里对 Gemini 再读 WIKONTIC_GEMINI_REASONING_EFFORT，默认 low
        self._gemini_reasoning_effort = gemini_reasoning_effort
        self.messages = []
        self.prompt_tokens_num = 0
        self.completion_tokens_num = 0
        self.current_cost = 0

        self._refine_attempt = 0
        self._prev_error = None  # store previous exception
        self.MAX_ATTEMPTS = max_attempts

        # Set pricing
        if model not in self.MODEL_PRICES:
            raise ValueError(f"Unknown model: {model}")
        self.input_price = self.MODEL_PRICES[model]["input"]
        self.output_price = self.MODEL_PRICES[model]["output"]

    def _repair_common_json_issues(self, text: str) -> str:
        """Repair a few recurrent LLM JSON formatting mistakes."""
        # Some model outputs emit a single qualifier object as:
        #   "qualifiers": ["relation": "...", "object": "..."]
        # Convert that into a valid one-element array of objects.
        text = re.sub(
            r'("qualifiers"\s*:\s*)\[\s*"relation"\s*:\s*(".*?"|null)\s*,\s*"object"\s*:\s*(".*?"|null)\s*\]',
            r'\1[{"relation": \2, "object": \3}]',
            text,
            flags=re.DOTALL,
        )
        # LLMs sometimes include raw LaTeX-like fragments such as \displaystyle
        # inside JSON strings. Escape backslashes that are not valid JSON escapes.
        text = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)
        # Remove trailing commas before a closing brace/bracket.
        text = re.sub(r",(\s*[}\]])", r"\1", text)
        return text

    def extract_json(self, text: str) -> Union[dict, list, str]:
        """Extract JSON from text, handling both code blocks and inline JSON."""
        patterns = [
            r"```json\s*(\{.*?\}|\[.*?\])\s*```",  # JSON in code blocks
            r"(\{.*?\}|\[.*?\])",  # Inline JSON
        ]

        candidates = [text, self._repair_common_json_issues(text)]
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                candidate = match.group(1)
                repaired_candidate = self._repair_common_json_issues(candidate)
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    try:
                        return json.loads(repaired_candidate)
                    except json.JSONDecodeError:
                        logger.error("Failed to parse JSON: %s", text)

        return text

    @retry(
        wait=wait_random_exponential(multiplier=1, max=60),
        before_sleep=before_sleep_log(logger, logging.ERROR),
        # stop=stop_after_attempt(5),
    )
    def get_completion(
        self, system_prompt: str, user_prompt: str, transform_to_json: bool = True
    ) -> Union[dict, list, str]:
        """Get completion from OpenAI API with retry logic."""
        if self.model in (
            "qwen/qwen3-32b",
            "Qwen/Qwen3-32B",
            "qwen32b",
        ):
            user_prompt = "/no_think \n" + user_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        create_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }
        # OpenAI 兼容网关（含常见 Gemini 代理）用 reasoning_effort 控制思考强度；Gemini 默认压低为 low
        if "gemini" in (self.model or "").lower():
            if self._gemini_reasoning_effort is not None:
                _eff = str(self._gemini_reasoning_effort).strip()
            else:
                _eff = os.getenv("WIKONTIC_GEMINI_REASONING_EFFORT", "low").strip()
            if _eff:
                create_kwargs["reasoning_effort"] = _eff

        response = self.client.chat.completions.create(**create_kwargs)
        usage = response.usage
        delta_prompt = getattr(usage, "prompt_tokens", None) or 0
        delta_completion = getattr(usage, "completion_tokens", None) or 0
        self.completion_tokens_num += delta_completion
        self.prompt_tokens_num += delta_prompt
        self.current_cost += (
            delta_completion * self.output_price + delta_prompt * self.input_price
        )

        content = response.choices[0].message.content
        if content is None:
            content = ""
        else:
            content = content.strip()

        # 默认关闭：避免巨量 request/response 刷屏与落盘撑满磁盘。
        # 开启逐条控制台日志：WIKONTIC_LOG_EACH_API_CALL=1
        _log_each = os.getenv("WIKONTIC_LOG_EACH_API_CALL", "0").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        # 仅当显式设置 WIKONTIC_API_JSON_LOG_DIR 时才落盘单次调用的 JSON bundle（不再默认写 cwd/wikontic_api_json_logs）
        _json_dir = os.getenv("WIKONTIC_API_JSON_LOG_DIR", "").strip()

        # JSON 落盘：完整 request + response（与网关实际结构一致，可为 chat.completion 或你们定义的 response 形态）
        if _json_dir:
            try:
                _extra = {
                    k: v
                    for k, v in create_kwargs.items()
                    if k not in ("model", "messages", "temperature")
                }
                _p = _write_wikontic_api_json_bundle(
                    _json_dir,
                    model=self.model,
                    messages=messages,
                    response=response,
                    usage=usage,
                    assistant_content_raw=content,
                    extra_request_fields=_extra or None,
                )
                logger.info("[openai_utils] api_json_bundle -> %s", _p)
            except Exception:
                logger.exception("[openai_utils] api_json_bundle write failed dir=%s", _json_dir)

        # 每次 chat 调用：控制台分块打印（开启：WIKONTIC_LOG_EACH_API_CALL=1）
        if _log_each:
            delta_usd = (
                delta_completion * self.output_price + delta_prompt * self.input_price
            ) / 1e6
            cum_usd = self.current_cost / 1e6
            total_tok = getattr(usage, "total_tokens", None)
            logger.info("=" * 80)
            logger.info(
                "[openai_utils] get_completion | model=%s | +prompt_tokens=%s +completion_tokens=%s "
                "+total_tokens=%s | this_call_usd=%.6f cum_usd=%.6f | cum_prompt_tokens=%s cum_completion_tokens=%s",
                self.model,
                delta_prompt,
                delta_completion,
                total_tok,
                delta_usd,
                cum_usd,
                self.prompt_tokens_num,
                self.completion_tokens_num,
            )
            _log_long("[openai_utils] REQUEST_MESSAGES_JSON", messages)
            try:
                if usage is not None and hasattr(usage, "model_dump"):
                    _log_long("[openai_utils] RESPONSE_USAGE_JSON", usage.model_dump())
                else:
                    _log_long("[openai_utils] RESPONSE_USAGE_REPR", repr(usage))
            except Exception:
                logger.exception("[openai_utils] usage dump failed")
            try:
                _log_long("[openai_utils] FULL_RESPONSE_JSON", response.model_dump())
            except Exception:
                _log_long("[openai_utils] FULL_RESPONSE_REPR", repr(response))
            msg = response.choices[0].message
            try:
                _log_long("[openai_utils] CHOICE0_MESSAGE_JSON", msg.model_dump())
            except Exception:
                logger.exception("[openai_utils] CHOICE0_MESSAGE dump failed")
            _log_long("[openai_utils] ASSISTANT_content_raw", content)
            rc = getattr(msg, "reasoning_content", None)
            if rc is not None and rc != "":
                _log_long("[openai_utils] ASSISTANT_reasoning_content", str(rc))
            else:
                logger.info("[openai_utils] ASSISTANT_reasoning_content=<None or empty>")
            logger.info("=" * 80)

        output = self.extract_json(content) if transform_to_json else content

        self.messages = messages + [{"role": "assistant", "content": output}]
        return output

    @tenacity.retry(stop=tenacity.stop_after_attempt(MAX_ATTEMPTS), reraise=True)
    def extract_triplets_from_text(self, text: str) -> dict:
        """Extract knowledge graph triplets from text."""

        self._refine_attempt += 1
        attempt = self._refine_attempt
        logger.log(
            logging.DEBUG,
            "Attempt of a function call extract_triplets_from_text: %s",
            attempt,
        )
        system_prompt = self.prompts["triplet_extraction"]
        if attempt > 1:
            prev_error = self._prev_error
            system_prompt += f"\n(Previous attempt #{attempt-1} failed with error: {prev_error}. Please adjust your answer!)"
            logger.log(logging.ERROR, "System prompt: %s", system_prompt)

        try:
            return self.get_completion(
                system_prompt=system_prompt, user_prompt=f'Text: "{text}"'
            )
        except Exception as e:
            self._prev_error = e
            # if json from output is broken after 3 attempts  - raise an exception
            logger.log(logging.ERROR, str(e))
            if attempt > self.MAX_ATTEMPTS:
                raise e

    @tenacity.retry(stop=tenacity.stop_after_attempt(MAX_ATTEMPTS), reraise=True)
    def refine_entity_types(
        self,
        text: str,
        triplet: dict,
        candidate_subject_types: List[str],
        candidate_object_types: List[str],
    ) -> dict:
        """Refine relations and entity types using candidate backbone triplets."""
        triplet_filtered = {
            k: triplet[k]
            for k in ["subject", "relation", "object", "subject_type", "object_type"]
        }

        candidates_subject_types_str = json.dumps(candidate_subject_types)
        candidates_object_types_str = json.dumps(candidate_object_types)
        logger.log(
            logging.DEBUG,
            "candidates subject types: %s\n%s",
            str(candidates_subject_types_str),
            "-" * 100,
        )
        logger.log(
            logging.DEBUG,
            "candidates object types: %s\n%s",
            str(candidates_object_types_str),
            "-" * 100,
        )

        self._refine_attempt += 1
        attempt = self._refine_attempt
        logger.log(
            logging.DEBUG, "Attempt of a function call refine_entity_types: %s", attempt
        )
        system_prompt = self.prompts["entity_types_ranker"]
        if attempt > 1:
            prev_error = self._prev_error
            system_prompt += f"\n(Previous attempt #{attempt-1} failed with error: {prev_error}. Please adjust your answer!)"
            logger.log(logging.ERROR, "System prompt: %s", system_prompt)

        try:
            output = self.get_completion(
                system_prompt=system_prompt,
                user_prompt=f'Text: "{text}\nExtracted Triplet: {json.dumps(triplet_filtered)}\n'
                f"Candidate Subject Types: {candidates_subject_types_str}\n"
                f"Candidate Object Types: {candidates_object_types_str}",
            )
        except Exception as e:
            self._prev_error = e
            logger.log(logging.ERROR, str(e))
            # if json from output is broken after 3 attempts  - raise an exception
            if attempt > self.MAX_ATTEMPTS:
                raise e

        logger.log(
            logging.DEBUG,
            "refined subject type: %s\n%s",
            str(output["subject_type"]),
            "-" * 100,
        )
        logger.log(
            logging.DEBUG,
            "refined object type: %s\n%s",
            str(output["object_type"]),
            "-" * 100,
        )

        try:
            assert (
                output["subject_type"] in candidate_subject_types
            ), "Refined subject type is not in candidate subject types"
            assert (
                output["object_type"] in candidate_object_types
            ), "Refined object type is not in candidate object types"
        except Exception as e:
            self._prev_error = e
            logger.log(logging.INFO, str(e))
            # do not raise an exception - save triplet in ontology filtered collection
        return output

    @tenacity.retry(stop=tenacity.stop_after_attempt(MAX_ATTEMPTS), reraise=True)
    def refine_relation(
        self, text: str, triplet: dict, candidate_relations: List[dict]
    ) -> dict:
        """Refine relation using candidate relations."""
        triplet_filtered = {
            k: triplet[k]
            for k in ["subject", "relation", "object", "subject_type", "object_type"]
        }

        candidates_str = json.dumps(candidate_relations, ensure_ascii=False)
        logger.log(
            logging.DEBUG,
            "candidates relations: %s\n%s",
            str(candidates_str),
            "-" * 100,
        )
        self._refine_attempt += 1
        attempt = self._refine_attempt

        logger.log(
            logging.DEBUG, "Attempt of a function call refine_relation: %s", attempt
        )
        system_prompt = self.prompts["relation_ranker"]

        if attempt > 1:
            prev_error = self._prev_error
            system_prompt += f"\n(Previous attempt #{attempt-1} failed with error {prev_error}. Please adjust your answer!)"
            logger.log(logging.ERROR, "System prompt: %s", system_prompt)
        try:
            output = self.get_completion(
                system_prompt=system_prompt,
                user_prompt=f'Text: "{text}\nExtracted Triplet: {json.dumps(triplet_filtered, ensure_ascii=False)}\n'
                f"Candidate relations: {candidates_str}",
                transform_to_json=True,
            )
        except Exception as e:
            self._prev_error = e
            logger.log(logging.ERROR, str(e))
            # if json from output is broken after 3 attempts  - raise an exception
            if attempt > self.MAX_ATTEMPTS:
                raise e

        logger.log(
            logging.DEBUG,
            "refined relation: %s\n%s",
            str(output["relation"]),
            "-" * 100,
        )

        try:
            assert (
                output["relation"] in candidate_relations
            ), "Refined relation is not in candidate relations"
        except Exception as e:
            self._prev_error = e
            logger.log(logging.INFO, str(e))
            # do not raise an exception - save triplet in ontology filtered collection

        return output

    @tenacity.retry(stop=tenacity.stop_after_attempt(MAX_ATTEMPTS), reraise=True)
    def refine_relation_wo_entity_types(
        self, text: str, triplet: dict, candidate_relations: List[dict]
    ) -> dict:
        """Refine relation using candidate relations."""
        triplet_filtered = {k: triplet[k] for k in ["subject", "relation", "object"]}
        candidates_str = json.dumps(candidate_relations, ensure_ascii=False)
        logger.log(
            logging.DEBUG,
            "candidates relations: %s\n%s",
            str(candidates_str),
            "-" * 100,
        )

        attempt = self._refine_attempt

        logger.log(
            logging.DEBUG,
            "Attempt of a function call refine_relation_wo_entity_types: %s",
            attempt,
        )
        self._refine_attempt += 1
        system_prompt = self.prompts["relation_ranker_wo_entity_types"]

        if attempt > 1:
            prev_error = self._prev_error
            system_prompt += f"\n(Previous attempt #{attempt-1} failed with error {prev_error}. Please adjust your answer!)"
            logger.log(logging.ERROR, "System prompt: %s", system_prompt)
        try:
            return self.get_completion(
                system_prompt=system_prompt,
                user_prompt=f'Text: "{text}\nExtracted Triplet: {json.dumps(triplet_filtered, ensure_ascii=False)}\n'
                f"Candidate relations: {candidates_str}",
                transform_to_json=False,
            )
        except Exception as e:
            self._prev_error = e
            logger.log(logging.ERROR, str(e))
            # if json from output is broken after 3 attempts  - raise an exception
            if self._refine_attempt > self.MAX_ATTEMPTS:
                raise e

    def refine_relation_and_entity_types(
        self, text: str, triplet: dict, candidate_triplets: List[dict]
    ) -> dict:
        """Refine relations and entity types using candidate backbone triplets."""
        triplet_filtered = {
            k: triplet[k]
            for k in ["subject", "relation", "object", "subject_type", "object_type"]
        }

        candidates_str = "".join(f"{json.dumps(c)}\n" for c in candidate_triplets)

        return self.get_completion(
            system_prompt=self.prompts["relation_entity_types_ranker"],
            user_prompt=f'Text: "{text}\nExtracted Triplet: {json.dumps(triplet_filtered)}\n'
            f"Candidate Triplets: {candidates_str}",
        )

    def refine_entity(
        self,
        text: str,
        triplet: dict,
        candidates: List[str],
        is_object: bool = False,
        role: str = "user",
    ) -> dict:
        """Refine subject/object names using candidate options from pre-built KG."""

        triplet_filtered = {k: triplet[k] for k in ["subject", "relation", "object"]}
        original_name = triplet_filtered["object" if is_object else "subject"]

        self._refine_attempt += 1
        attempt = self._refine_attempt

        logger.log(
            logging.DEBUG, "Attempt of a function call refine_entity: %s", attempt
        )
        prompt_key = "object_ranker" if is_object else "subject_ranker"
        entity_type = "Object" if is_object else "Subject"
        system_prompt = self.prompts[prompt_key]

        if attempt > 1:
            prev_error = self._prev_error
            system_prompt += f"\n(Previous attempt #{attempt-1} failed with error: {prev_error}. Please adjust your answer!)"
            logger.log(logging.ERROR, "System prompt: %s", system_prompt)

        try:
            return self.get_completion(
                system_prompt=system_prompt,
                user_prompt=f'Text: "{text}\nRole: {role}\nExtracted Triplet: {json.dumps(triplet_filtered, ensure_ascii=False)}\n'
                f"Original {entity_type}: {original_name}\n"
                f'Candidate {entity_type}s: {json.dumps(candidates, ensure_ascii=False)}"',
                transform_to_json=False,
            )
        except Exception as e:
            self._prev_error = e
            logger.log(logging.ERROR, str(e))
            # if json from output is broken after 3 attempts  - raise an exception
            if attempt > self.MAX_ATTEMPTS:
                raise e

    def extract_entities_from_question(self, question: str) -> dict:
        """Extract entities from a question."""
        return self.get_completion(
            system_prompt=self.prompts["question_entity_extractor"],
            user_prompt=f"Question: {question}",
        )

    def identify_relevant_entities(
        self, question: str, entity_list: List[str]
    ) -> List[str]:
        """Identify entities relevant to a question."""
        return self.get_completion(
            system_prompt=self.prompts["question_entity_ranker"],
            user_prompt=f"Question: {question}\nEntities: {entity_list}",
        )

    def identify_relevant_entities_wo_types(
        self, question: str, entity_list: List[str]
    ) -> List[str]:
        """Identify entities relevant to a question."""
        return self.get_completion(
            system_prompt=self.prompts["question_entity_ranker_wo_types"],
            user_prompt=f"Question: {question}\nEntities: {entity_list}",
        )

    def answer_question(self, question: str, triplets: List[dict]) -> str:
        """Answer a question using knowledge graph triplets."""
        return self.get_completion(
            system_prompt=self.prompts["qa"],
            user_prompt=f'Question: {question}\n\nTriplets: "{triplets}"',
            transform_to_json=False,
        )

    def collapse_question(
        self, original_question: str, question: str, answer: str
    ) -> str:
        """Collapse a question using knowledge graph triplets."""
        return self.get_completion(
            system_prompt=self.prompts["qa_collapsing"],
            user_prompt=f"Original multi-hop question: {original_question}\n\Answered sub-question: {question}\n\Answer: {answer}",
            transform_to_json=True,
        )

    def decompose_question(self, question: str) -> str:
        """Decompose a question using knowledge graph triplets."""
        return self.get_completion(
            system_prompt=self.prompts["question_decomposition_1"],
            user_prompt=f"Question: {question}",
            transform_to_json=False,
        )

    def check_if_question_is_answered(
        self, question: str, subquestions: List[str], answers: List[str]
    ) -> str:
        """Check if a question is answered."""
        user_prompt = (
            f"Original multi-hop question: {question}\nQuestion->answer sequence:\n"
        )
        for question, answer in zip(subquestions, answers):
            user_prompt += f"{question} -> {answer}\n"
        return self.get_completion(
            system_prompt=self.prompts["qa_is_answered"],
            user_prompt=user_prompt,
            transform_to_json=False,
        )

    def calculate_cost(self) -> float:
        """Calculate the total cost of API usage."""
        return self.current_cost / 1e6

    def calculate_used_tokens(self) -> int:
        """Calculate the total # of used tokens for generation"""
        return self.prompt_tokens_num, self.completion_tokens_num

    def reset_tokens(self):
        """Reset the total # of used tokens for generation"""
        self.prompt_tokens_num = 0
        self.completion_tokens_num = 0

    def reset_messages(self):
        """Reset the messages"""
        self.messages = []

    def reset_error_state(self):
        self._prev_error = None
        self._refine_attempt = 0
