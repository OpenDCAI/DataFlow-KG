import textwrap
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC
from typing import List, Dict, Any


@PROMPT_REGISTRY.register()
class MMKGVisualTripleExtractionPrompt(PromptABC):

    def __init__(self, lang: str = "en"):
        self.lang = lang

    def build_system_prompt(self) -> str:

        if self.lang == "zh":
            return textwrap.dedent("""
            你是一个视觉理解与知识图谱构建专家。

            任务：
            根据图像内容，从给定实体列表中识别哪些实体真实出现在图像中。

            规则：
            1. 只能基于图像内容判断
            2. 只能从给定实体列表中选择
            3. 不允许生成新的实体
            4. 如果没有匹配实体返回空列表
            5. 一张图像可能包含多个实体

            输出：
            仅返回 JSON
            """)
        else:
            return textwrap.dedent("""
            You are an expert in visual understanding and knowledge graph construction.

            Task:
            Identify which entities from a provided list appear in the image.

            Rules:
            - Only use visible information from the image
            - Entities MUST come from the given list
            - Do NOT invent new entities
            - If none match, return an empty list
            - Multiple entities may appear

            Return JSON only.
            """)

    def build_prompt(self, img_id: str, entity_list: list[str]) -> str:

        entity_text = ", ".join(entity_list)

        if self.lang == "zh":

            return textwrap.dedent(f"""
            图像ID: {img_id}

            实体列表：
            {entity_text}

            任务：
            判断图像中出现了哪些实体。

            评分标准：
            0-3 图像模糊
            4-6 不确定
            7-10 明确匹配

            输出 JSON:

            {{
              "entity": ["实体1","实体2"],
              "quality_score": 0
            }}

            只返回JSON。
            """)

        else:

            return textwrap.dedent(f"""
            Image ID: {img_id}

            Entity list:
            {entity_text}

            Task:
            Identify which entities from the list are visually present in the image.

            Quality score:
            0-3  : image unclear
            4-6  : uncertain
            7-10 : confident match

            Output JSON:

            {{
              "entity": ["entity1","entity2"],
              "quality_score": 0
            }}

            Return JSON only.
            """)



@PROMPT_REGISTRY.register()
class MMKGSubgraphBaseQAGenerationPrompt(PromptABC):

    def __init__(self, lang: str = "en"):
        self.lang = lang

    def build_system_prompt(self) -> str:
        if self.lang == "zh":
            return textwrap.dedent("""
            你是多模态知识图谱问答专家，基于子图和视觉信息生成问答对。

            核心任务：
            生成需要同时依赖子图关系和视觉信息的问答对。

            关键要求：
            1. **KG推理优先**：问答必须基于给定的子图关系进行推理
            2. **视觉结合**：以视觉实体为起点，通过子图关系推导答案
            3. **约束**：只能使用子图中明确给出的实体和关系

            禁止的问题类型：
            - 纯视觉问题："图像中的X是什么颜色？"
            - 纯知识问题："子图中A与B的关系是什么？"
            - 子图外信息：任何不在给定子图中的内容

            输出格式：
            仅返回JSON，无其他内容。
            """)
        else:
            return textwrap.dedent("""
            You are a multimodal knowledge graph QA expert, generating QA pairs based on subgraphs and visual information.

            Core Task:
            Generate QA pairs that require both subgraph relationships and visual information.

            Key Requirements:
            1. **KG Reasoning First**: QA must be based on reasoning through given subgraph relationships
            2. **Visual Integration**: Start from visual entities, derive answers through subgraph relationships
            3. **Constraints**: Only use entities and relationships explicitly given in the subgraph

            Prohibited Question Types:
            - Pure visual questions: "What color is X in the image?"
            - Pure knowledge questions: "What is the relationship between A and B in the subgraph?"
            - External information: Any content not in the given subgraph

            Output Format:
            Return ONLY JSON, no other content.
            """)

    def build_prompt(self, img_id: str, subgraph: List[str], vis_triple: List[str]) -> str:
        """
        img_id: 图片ID (比如 img_canada_01)
        subgraph: 子图列表
        vis_triple: 对应图片的视觉三元组
        """
        subgraph_text = "\n".join(subgraph)
        vis_text = "\n".join([t for t in vis_triple if img_id in t])

        if self.lang == "zh":
            return textwrap.dedent(f"""
            图像ID: {img_id}

            子图网络（关系网络）：
            {subgraph_text}

            视觉锚点（视觉事实）：
            {vis_text}

            网络推理任务执行：

            **任务执行步骤**：
            1. **检查视觉实体**：确认视觉三元组中的实体是否在子图关系中出现（作为主语或宾语）
            2. **生成QA对**：
               - 如果实体不在子图中：生成报错QA 
               {{"question": "[实体名]实体错误", "answer": "视觉实体[实体名]不存在于给定子图中"}}
               - 如果实体在子图中：基于该实体的子图关系生成2-3个推理QA

            **KG推理QA要求**：
            - 从视觉实体出发，通过子图关系推导其他信息
            - 答案必须完全基于给定的子图关系
            - 问题要体现推理过程，不能是简单的关系查询

            **约束**：
            - 所有答案必须严格基于给定的子图关系
            - 绝对禁止使用子图外的任何信息
            - 必须输出QA对：包括正确QA和错误QA

            **示例1**：
            子图关系：
            <subj> Lucy <obj> University Toronto <rel> studies_at
            <subj> Lucy <obj> Clean Earth <rel> joins
            <subj> Henry <obj> Lucy <rel> is_inspired_by

            视觉锚点：
            <subj> University Toronto <rel> depicted_in <obj> img_University_Toronto_02

            输出：
            {{
              "QA_pairs": [
                {{"question": "基于图像img_University_Toronto_02，在这所大学学习的人还参与了哪些组织？", "answer": "图像img_University_Toronto_02中是University Toronto，Lucy在该大学学习，同时参与了Clean Earth组织。"}},
                {{"question": "基于图像img_University_Toronto_02，这所大学如何通过子图关系连接到环保活动？", "answer": "图像img_University_Toronto_02中是University Toronto，通过Lucy连接到环保活动，Lucy在该大学学习并加入了Clean Earth。"}}
              ]
            }}
            

            **示例2**：
            子图关系：
            <subj> Henry <obj> Maple Leaves <rel> forms
            <subj> Maple Leaves <obj> Polar Lights <rel> releases
            <subj> Maple Leaves <obj> Paris <rel> performs_in

            视觉锚点：
            <subj> Maple Leaves <rel> depicted_in <obj> img_canada_01

            输出：
            {{
              "QA_pairs": [
                {{"question": "基于图像img_canada_01，这个乐队发布了什么作品？", "answer": "图像img_canada_01中是Maple Leaves乐队的标志，根据子图，Maple Leaves发布了Polar Lights。"}},
                {{"question": "基于图像img_canada_01，这个乐队在哪里演出？", "answer": "图像img_canada_01中是Maple Leaves乐队的标志，根据子图，Maple Leaves在Paris演出。"}}
              ]
            }}

            输出格式：
            {{
              "QA_pairs": [
                  {{"question": "问题1", "answer": "答案1"}},
                  {{"question": "问题2", "answer": "答案2"}}
              ]
            }}
            """)
        else:
            return textwrap.dedent(f"""
            Image ID: {img_id}

            Subgraph Network (Relationship Network):
            {subgraph_text}

            Visual Anchors (Visual Facts):
            {vis_text}

            Network Reasoning Task Execution:

            **Task Execution Steps**:
            1. **Check Visual Entities**: Confirm if entities in visual triples appear in subgraph relationships (as subject or object)
            2. **Generate QA Pairs**:
               - If entity not in subgraph: Generate error QA {{"question": "[entity_name] entity error", "answer": "Visual entity [entity_name] does not exist in the given subgraph"}}
               - If entity in subgraph: Generate 2-3 reasoning QA based on that entity's subgraph relationships

            **KG Reasoning QA Requirements**:
            - Start from visual entities, derive other information through subgraph relationships
            - Answers must be completely based on given subgraph relationships
            - Questions should reflect reasoning process, not simple relationship queries

            **Constraints**:
            - All answers must be strictly based on given subgraph relationships
            - Absolutely prohibit using any information outside the subgraph
            - Must output QA pairs: including correct QA and incorrect QA

            **Example 1**:
            Subgraph Relations:
            <subj> Lucy <obj> University Toronto <rel> studies_at
            <subj> Lucy <obj> Clean Earth <rel> joins
            <subj> Henry <obj> Lucy <rel> is_inspired_by

            Visual Anchors:
            <subj> University Toronto <rel> depicted_in <obj> img_University_Toronto_02

            Output:
            {{
              "QA_pairs": [
                {{"question": "Based on the image img_University_Toronto_02, what organizations do people studying at this university participate in?", "answer": "The image img_University_Toronto_02 shows University Toronto, Lucy studies at this university and also participates in the Clean Earth organization."}},
                {{"question": "Based on the image img_University_Toronto_02, how does this university connect to environmental activities through the subgraph?", "answer": "The image img_University_Toronto_02 shows University Toronto, which connects to environmental activities through Lucy, who studies there and joins Clean Earth."}}
              ]
            }}

            

            **Example 2**:
            Subgraph Relations:
            <subj> Henry <obj> Maple Leaves <rel> forms
            <subj> Maple Leaves <obj> Polar Lights <rel> releases
            <subj> Maple Leaves <obj> Paris <rel> performs_in

            Visual Anchors:
            <subj> Maple Leaves <rel> depicted_in <obj> img_canada_01

            Output:
            {{
              "QA_pairs": [
                {{"question": "Based on the image img_University_Toronto_02, what did this band release?", "answer": "The image img_University_Toronto_02 shows the Maple Leaves band symbol, according to the subgraph, Maple Leaves released Polar Lights."}},
                {{"question": "Based on the image img_University_Toronto_02, where does this band perform?", "answer": "The image img_University_Toronto_02 shows the Maple Leaves band symbol, according to the subgraph, Maple Leaves performs in Paris."}}
              ]
            }}

            Output Format:
            {{
              "QA_pairs": [
                  {{"question": "Question 1", "answer": "Answer 1"}},
                  {{"question": "Question 2", "answer": "Answer 2"}}
              ]
            }}
            """)


@PROMPT_REGISTRY.register()
class MMKGPathBasedQAGenerationPrompt(PromptABC):

    def __init__(self, lang: str = "en"):
        self.lang = lang

    def build_system_prompt(self) -> str:
        if self.lang == "zh":
            return textwrap.dedent("""
            你是多模态知识图谱问答专家，基于路径和视觉信息生成问答对。

            核心任务：
            生成需要完整路径推理的问答对。

            关键要求：
            1. 问答必须需要完整的A→B→...→C路径推理才能回答
            2. 从视觉信息出发，通过路径推导最终答案
            3. 只能使用路径中明确给出的实体和关系

            输出格式：
            仅返回JSON，无其他内容。
            """)
        return textwrap.dedent("""
        You are a multimodal knowledge graph QA expert, generating QA pairs based on paths and visual information.

        Core Task:
        Generate QA pairs that require complete path reasoning.

        Key Requirements:
        1. QA must require complete A→B→...→C path reasoning to answer
        2. Start from visual information, derive final answers through path reasoning
        3. Only use entities and relationships explicitly given in the path

        Output Format:
        Return ONLY JSON, no other content.
        """)

    def build_prompt(self, img_id: str, path: str, vis_triple: List[str]) -> str:
        vis_text = "\n".join([t for t in vis_triple if img_id in t])

        if self.lang == "zh":
            return textwrap.dedent(f"""
            图像ID: {img_id}

            知识图谱路径：
            {path}

            视觉锚点：
            {vis_text if vis_text else "无"}

            任务：
            如果无视觉锚点，返回空数组。
            如果有视觉锚点，生成1-2个需要完整路径推理的问答对。

            **示例1**：
            路径：Henry forms Maple Leaves || Maple Leaves releases Polar Lights
            视觉锚点：<subj> Maple Leaves <rel> depicted_in <obj> img_canada_01

            输出：
            {{
              "QA_pairs": [
                {{"question": "基于图像img_canada_01，这个实体的创建者发布了什么作品？", "answer": "图像img_canada_01中是Maple Leaves，根据路径Henry forms Maple Leaves，然后Maple Leaves releases Polar Lights，所以Henry通过Maple Leaves发布了Polar Lights。"}},
                {{"question": "基于图像img_canada_01，什么作品是通过这个乐队发布的？", "answer": "图像img_canada_01中是Maple Leaves，根据路径Maple Leaves releases Polar Lights，所以通过Maple Leaves发布的作品是Polar Lights。"}}
              ]
            }}

            **示例2（时序推理）**：
            路径：Maple Leaves releases Polar Lights || Polar Lights is_released_on August 12 2020
            视觉锚点：<subj> Maple Leaves <rel> depicted_in <obj> img_canada_01

            输出：
            {{
              "QA_pairs": [
                {{"question": "基于图像img_canada_01，这个乐队的作品是什么时候发布的？", "answer": "图像img_canada_01中是Maple Leaves，根据路径Maple Leaves releases Polar Lights，然后Polar Lights is_released_on August 12 2020，所以这个乐队的作品在2020年8月12日发布。"}}
              ]
            }}

            """)
        return textwrap.dedent(f"""
        Image ID: {img_id}

        Knowledge Graph Path:
        {path}

        Visual Anchors:
        {vis_text if vis_text else "None"}

        Task:
        If no visual anchors, return empty array.
        If visual anchors exist, generate 1-2 QA pairs requiring complete path reasoning.

        **Example 1**:
        Path: Henry forms Maple Leaves || Maple Leaves releases Polar Lights
        Visual Anchors: <subj> Maple Leaves <rel> depicted_in <obj> img_canada_01

        Output:
        {{
          "QA_pairs": [
            {{"question": "Based on the image img_canada_01, what work did the creator of this entity release?", "answer": "The image img_canada_01 shows Maple Leaves, according to path Henry forms Maple Leaves, then Maple Leaves releases Polar Lights, so Henry released Polar Lights through Maple Leaves."}},
            {{"question": "Based on the image img_canada_01, what work was released through this band?", "answer": "The image img_canada_01 shows Maple Leaves, according to path Maple Leaves releases Polar Lights, so the work released through Maple Leaves is Polar Lights."}}
          ]
        }}

        **Example 2 (Temporal reasoning)**:
        Path: Maple Leaves releases Polar Lights || Polar Lights is_released_on August 12 2020
        Visual Anchors: <subj> Maple Leaves <rel> depicted_in <obj> img_canada_01

        Output:
        {{
          "QA_pairs": [
            {{"question": "Based on the image img_canada_01, when was this band's work released?", "answer": "The image img_canada_01 shows Maple Leaves, according to path Maple Leaves releases Polar Lights, then Polar Lights is_released_on August 12 2020, so this band's work was released on August 12, 2020."}}
          ]
        }}


        Output Format:
        {{
          "QA_pairs": [
              {{"question": "Question 1", "answer": "Answer 1"}},
              {{"question": "Question 2", "answer": "Answer 2"}}
          ]
        }}
        """)


@PROMPT_REGISTRY.register()  # type: ignore
class MMKGTextTripleVerificationPrompt(PromptABC):
    
    def __init__(self, lang: str = "en"):
        self.lang = lang
    
    def build_system_prompt(self) -> str:
        """Build system prompt for text triple verification."""
        if self.lang == "zh":
            return textwrap.dedent("""\
                你是一个严格的知识图谱审计员。
                你的任务是验证给定的三元组是否被上下文明确支持。
                
                判断标准：
                1. 忽略外部知识，只使用提供的上下文
                2. 如果三元组与文本矛盾或未被提及，返回 FALSE
                3. 如果三元组被支持，返回 TRUE
                
                输出格式：
                返回 JSON 对象：{"reasoning": "...", "verdict": true/false}
            """)
        else:
            return textwrap.dedent("""\
                You are a strict knowledge graph auditor.
                Your task is to verify if the following Triple is explicitly supported by the Context.
                
                Requirements:
                1. Ignore external knowledge. Only use the Context.
                2. If the triple contradicts the text or is not mentioned, return FALSE.
                3. If it is supported, return TRUE.
                
                Output format JSON: {"reasoning": "...", "verdict": true/false}
            """)
    
    def build_prompt(self, context: str, subject: str, relation: str, obj: str) -> str:
        """Build user prompt for text triple verification."""
        if self.lang == "zh":
            return textwrap.dedent(f"""\
                上下文："{context}"
                
                三元组：
                - 主体：{subject}
                - 关系/属性：{relation}
                - 客体/值：{obj}
                
                请判断这个三元组是否被上下文支持。
                返回 JSON 对象：{{"reasoning": "...", "verdict": true/false}}
            """)
        else:
            return textwrap.dedent(f"""\
                Context: "{context}"
                
                Triple:
                - Subject: {subject}
                - Relation/Attribute: {relation}
                - Object/Value: {obj}
                
                Please determine if this triple is supported by the context.
                Output format JSON: {{"reasoning": "...", "verdict": true/false}}
            """)


@PROMPT_REGISTRY.register()  # type: ignore
class MMKGVisualTripleVerificationPrompt(PromptABC):
    
    def __init__(self, lang: str = "en"):
        self.lang = lang
    
    def build_system_prompt(self) -> str:
        """Build system prompt for visual triple verification."""
        if self.lang == "zh":
            return textwrap.dedent("""\
                你是一个知识图谱的视觉审计员。
                你的任务是验证图片是否描绘了指定的实体。
                
                判断标准：
                1. 仔细观察图片内容
                2. 判断图片中是否包含或代表了指定的实体
                3. 使用三值判断：
                   - true: 确认图片包含该实体
                   - false: 确认图片不包含该实体（明显错误/幻觉）
                   - uncertain: 无法确定（图片模糊、信息不足、但有相关特征）
                4. 给出置信度分数（0.0-1.0）
                
                重要提示：
                - 只有当你确信图片与实体完全无关时，才返回 false
                - 如果图片质量差但有相关特征，应返回 uncertain
                - uncertain 不会被计入幻觉率
                
                输出格式：
                返回 JSON 对象：{"reasoning": "...", "verdict": "true/false/uncertain", "confidence": 0.0-1.0}
            """)
        else:
            return textwrap.dedent("""\
                You are a visual auditor for a knowledge graph.
                Your task is to verify if the image depicts the specified entity.
                
                Evaluation Criteria:
                1. Carefully observe the image content
                2. Determine if the image contains or represents the specified entity
                3. Use three-value verdict:
                   - true: Confirmed the image contains the entity
                   - false: Confirmed the image does NOT contain the entity (clear hallucination)
                   - uncertain: Cannot determine (blurry image, insufficient info, but has related features)
                4. Provide a confidence score (0.0-1.0)
                
                Important:
                - Only return false when you are certain the image is completely unrelated to the entity
                - If image quality is poor but has related features, return uncertain
                - uncertain cases will NOT be counted as hallucinations
                
                Output format JSON: {"reasoning": "...", "verdict": "true/false/uncertain", "confidence": 0.0-1.0}
            """)
    
    def build_prompt(self, entity_name: str) -> str:
        """Build user prompt for visual triple verification."""
        if self.lang == "zh":
            return textwrap.dedent(f"""\
                三元组声明：这张图片与实体 "{entity_name}" 相关联。
                
                这张图片是否包含或代表 "{entity_name}"？
                
                请仔细分析后返回：
                - verdict: "true"（确认包含）/ "false"（确认不包含）/ "uncertain"（无法确定）
                - confidence: 置信度分数（0.0-1.0）
                - reasoning: 判断理由
                
                返回 JSON 对象：{{"reasoning": "...", "verdict": "true/false/uncertain", "confidence": 0.0-1.0}}
            """)
        else:
            return textwrap.dedent(f"""\
                Triple Claim: This image is linked to the entity "{entity_name}".
                
                Does this image contain or represent "{entity_name}"?
                
                Please analyze carefully and return:
                - verdict: "true" (confirmed) / "false" (confirmed NOT) / "uncertain" (cannot determine)
                - confidence: confidence score (0.0-1.0)
                - reasoning: explanation
                
                Output format JSON: {{"reasoning": "...", "verdict": "true/false/uncertain", "confidence": 0.0-1.0}}
            """)

