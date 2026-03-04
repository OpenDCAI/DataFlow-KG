from dataflow.core.prompt import PromptABC
from dataflow.utils.registry import PROMPT_REGISTRY
import json
from typing import Dict, List, Optional, Any


@PROMPT_REGISTRY.register()
class SparqlCandidateSelectionPrompt(PromptABC):
    """SPARQL候选选择的Prompt"""

    def __init__(self):
        pass

    def build_prompt(self, sparql_list: List[str], num_candidates: int) -> str:
        prompt = f"""你是一个知识图谱问答数据生成助手。我需要从多个 SPARQL 路径中选择一个最有价值的查询来生成QA对。
【输入信息】：
候选 SPARQL 列表（共 {num_candidates} 个）：
{sparql_list}
【选择标准】：
1. **语义合理性**：SPARQL 应体现清晰、合理的关系链条
2. **信息价值**：优先选择信息量更高、关系更有代表性的查询
3. **可问性**：基于这条 SPARQL 能够生成自然、明确的问题

请从候选 1 到候选 {num_candidates} 中选择一个最有价值的候选，并简要说明选择理由。
【输出格式】
严格按照以下JSON格式输出：
{{
    "selected_candidate": [候选编号，整数，范围1-{num_candidates}],
    "reason": "[选择理由，字符串]"
}}"""
        return prompt




@PROMPT_REGISTRY.register()
class SparqlReverseGeneratorPrompt(PromptABC):
    """SPARQL反向生成的Prompt"""
    
    def __init__(self):
        pass
    
    def build_prompt(
        self,
        question: str,
        answer_str: str,
        path_entities: List[str],
        path_relations: List[str],
        sparql_pattern: Optional[str] = None
    ) -> str:
        """构建SPARQL反向生成prompt
        
        Args:
            question: 自然语言问题
            answer_str: 答案实体字符串
            path_entities: 路径实体名称列表
            path_relations: 路径关系名称列表
            sparql_pattern: SPARQL模式（可选）
        """
        path_entities_str = json.dumps(path_entities, ensure_ascii=False, indent=2)
        path_relations_str = json.dumps(path_relations, ensure_ascii=False, indent=2)
        if sparql_pattern:
            prompt = f"""你的任务是将自然语言问题准确映射为可执行的 SPARQL 1.1 查询语句。
【输入信息】
1. 自然语言问题：{question}
2. 答案实体：{answer_str}
3. SPARQL模式约束（必须遵循此结构）：{sparql_pattern}
4. 可用的节点字典：{path_entities_str}
5. 可用的关系字典：{path_relations_str}

【操作步骤】
1. 语义映射：理解问题意图，从“可用节点名称”和“可用关系名称”中挑选出匹配的实体与关系。
2. 模式填充：只使用名称替换“SPARQL 模式约束”中的占位符。实体与关系都用 <名称> 形式输出。
3. 纯名称输出：禁止使用 QID/PID，也不要写 wd:/wdt: 前缀。

【严格约束】
1. 绝对忠实于SPARQL模式约束：仅进行占位符的精确替换，严禁增删改动 SPARQL 模式的基础结构（如变量名、括号、层级等）。
2. 纯净输出要求：只输出合法的 JSON 对象，绝对不要包含任何多余的解释说明、思考过程，也不要使用 ```json 这样的 Markdown 代码块标记。
3. 深呼吸，step by step地完成任务。

请按照以下严格的 JSON 格式输出：
{{"sparql_query": "完整的SPARQL查询语句"}}
"""
        return prompt

@PROMPT_REGISTRY.register()
class QuestionRewriterPrompt(PromptABC):
    """问题改写 Prompt：输入 QA 对，只改写 question，可带 answer 作上下文防语义偏移。"""

    def __init__(self):
        pass

    def build_prompt(self, question: str, answer: Any = None) -> str:
        return f'''

        【输入】：
        原句：{question}
        答案：{answer}

        【输出格式】：
        只输出 JSON 对象，格式如下：
        {{
            "question": "改写后的问题",
            "strategies": ["选择的改写策略","选择的改写策略2","..."]
        }}
        '''


@PROMPT_REGISTRY.register()
class SparqlCompletedQuestionPrompt(PromptABC):
    """基于完整 SPARQL 生成问题的 Prompt"""

    def __init__(self):
        pass

    def build_prompt(self, sparql_completed: str) -> str:
        return f"""你是一个专业的知识图谱问答数据生成器。你的任务是根据提供的包含语义信息的 SPARQL 查询结构，生成一个自然语言问题及其答案。该问题必须准确反映 SPARQL 查询的语义意图，明确指出目标实体和关系链。该问题不应包含任何中间实体的名称；仅使用起始实体和关系描述来隐含地表达路径。
【输入信息】：
包含语义信息的SPARQL结构：{sparql_completed}
【输出格式】
只输出JSON对象，格式如下：
{{
"question":"生成的自然语言问题"
}}。"""