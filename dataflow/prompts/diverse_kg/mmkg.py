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



class MMKGSubgraphBaseQAGenerationPrompt(PromptABC):

    def __init__(self, lang: str = "en"):
        self.lang = lang

    def build_system_prompt(self) -> str:
        if self.lang == "zh":
            return textwrap.dedent("""
            你是视觉理解与知识图谱领域的资深专家，需严格按照以下规则生成问答对。

            核心任务：
            基于图像的视觉内容和子图结构化信息，生成高质量的问答对（QA），且每个QA必须同时深度结合两类信息。

            严格规则：
            1. **核心强制要求**：所有问答对必须同时引用「子图中的实体/关系信息」和「图像中的视觉事实」，禁止仅基于图片视觉特征（如颜色、形状、位置、动作）生成无子图关联的QA（例如仅问“图片中物体的颜色是什么？”“XX在图片的哪个位置？”）；
            2. 问题必须明确涉及子图中的至少一个实体/关系，且需围绕该实体/关系结合图片视觉信息提问；
            3. 答案必须同时满足：① 符合子图的结构化知识逻辑；② 基于图片可见的视觉事实验证；
            4. 禁止生成子图中未包含的实体，禁止编造图像中未出现的视觉特征；
            5. 每张图片需生成2-5个不同类型的问答对（如事实型、推理型、关联型），避免重复，且所有QA需覆盖子图核心实体/关系；
            6. 问题需自然通顺、语义明确，答案需简洁准确、无歧义，禁止生成无意义或边缘的QA内容。

            输出要求：
            - 仅返回JSON格式，无任何额外文本、示例或解释；
            - JSON结构必须严格匹配指定格式，字段不可缺失、不可新增。
            """)
        else:
            return textwrap.dedent("""
            You are a senior expert in visual understanding and knowledge graphs, and must generate question-answer pairs (QA) in strict accordance with the following rules.

            Core Task:
            Generate high-quality question-answer pairs (QA) based on the visual content of the image and structured information of the subgraph, with each QA deeply combining both types of information simultaneously.

            Strict Rules:
            1. **Core Mandatory Requirement**: All QA pairs must reference both "entity/relation information in the subgraph" and "visual facts in the image" at the same time. It is prohibited to generate QA without subgraph association based solely on image visual features (e.g., color, shape, position, action) (e.g., only asking "What is the color of the object in the picture?" "Where is XX in the picture?");
            2. Questions must explicitly involve at least one entity/relation from the subgraph, and must ask questions combining image visual information around the entity/relation;
            3. Answers must simultaneously satisfy: ① Comply with the structured knowledge logic of the subgraph; ② Verified based on visible visual facts of the image;
            4. Prohibit generating entities not included in the subgraph, and prohibit inventing visual features not present in the image;
            5. Generate 2-5 different types of QA pairs per image (e.g., factual, inferential, associative), avoiding repetition, and all QA must cover core entities/relations in the subgraph;
            6. Questions must be natural, fluent and semantically clear, answers must be concise, accurate and unambiguous, and meaningless or marginal QA content is prohibited.

            Output Requirements:
            - Return ONLY JSON format, without any additional text, examples, or explanations;
            - The JSON structure must strictly match the specified format, with no missing or additional fields.
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

            子图三元组（结构化知识）：
            {subgraph_text}

            图像视觉三元组（视觉事实）：
            {vis_text}

            任务执行细则：
            1. 生成2-5个问答对，每个QA必须同时满足：
               - 问题中明确包含子图中的至少一个实体/关系；
               - 问题需围绕该实体/关系，结合图片视觉信息展开（而非仅问视觉特征）；
               - 答案需同时基于子图逻辑和图片视觉事实给出；
            2. 禁止生成仅询问图片视觉特征的QA（反例：“图片中的杯子是什么颜色？”）；
            3. 正确示例方向：
               - 子图：<咖啡杯, 所属场景, 咖啡店>；视觉三元组：<咖啡杯, 颜色, 白色>
               - 合规问题：“图片中属于咖啡店场景的咖啡杯是什么颜色？”
               - 合规答案：“白色”
            4. 所有QA需紧密关联子图核心逻辑，避免脱离子图的纯视觉问答。

            强制输出格式（仅返回此JSON）：
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

            Subgraph Triples (Structured Knowledge):
            {subgraph_text}

            Image Visual Triples (Visual Facts):
            {vis_text}

            Task Execution Details:
            1. Generate 2-5 QA pairs, each QA must simultaneously satisfy:
               - The question explicitly contains at least one entity/relation from the subgraph;
               - The question must be expanded around the entity/relation combined with image visual information (rather than only asking visual features);
               - The answer must be given based on both subgraph logic and image visual facts;
            2. Prohibit generating QA that only asks about image visual features (counterexample: "What color is the cup in the picture?");
            3. Correct example direction:
               - Subgraph: <Coffee cup, Belongs to scene, Coffee shop>; Visual triple: <Coffee cup, Color, White>
               - Compliant question: "What color is the coffee cup belonging to the coffee shop scene in the picture?"
               - Compliant answer: "White"
            4. All QA must be closely related to the core logic of the subgraph, avoiding pure visual QA that is divorced from the subgraph.

            Mandatory Output Format (return ONLY this JSON):
            {{
              "QA_pairs": [
                  {{"question": "Question 1", "answer": "Answer 1"}},
                  {{"question": "Question 2", "answer": "Answer 2"}}
              ]
            }}
            """)


class MMKGPathBasedQAGenerationPrompt(PromptABC):

    def __init__(self, lang: str = "en"):
        self.lang = lang

    def build_system_prompt(self) -> str:

        if self.lang == "zh":
            return textwrap.dedent("""
            你是视觉理解与知识图谱领域的资深专家，需严格按照以下规则生成问答对。

            核心任务：
            基于图像的视觉内容以及给定的知识图谱路径（path），生成高质量问答对（QA）。

            严格规则：
            1. 每个QA必须同时结合「路径中的实体/关系信息」和「图像中的视觉事实」；
            2. 问题必须围绕路径中的关系逻辑进行提问；
            3. 问题必须包含路径中的至少一个实体；
            4. 答案必须符合路径的知识逻辑，并能够通过图像视觉信息验证；
            5. 禁止生成路径中不存在的实体或关系；
            6. 禁止生成仅基于视觉特征的问题（例如颜色、位置、形状等）；
            7. 问题需自然流畅，答案需简洁准确；
            8. **如果没有图像或视觉三元组信息，则必须返回空结果。**

            输出要求：
            - 仅返回JSON
            - 不允许出现解释文本
            - JSON结构必须严格符合指定格式
            - 如果没有视觉信息，返回：
              {"QA_pairs": []}
            """)
        else:
            return textwrap.dedent("""
            You are a senior expert in visual understanding and knowledge graphs.

            Core Task:
            Generate high-quality question-answer pairs (QA) based on the visual content of the image and the given knowledge graph path.

            Strict Rules:
            1. Each QA pair must combine both the "entity/relation information in the path" and the "visual facts in the image";
            2. Questions must be based on the relational logic of the path;
            3. The question must contain at least one entity from the path;
            4. The answer must follow the knowledge logic of the path and be verifiable using visual evidence from the image;
            5. Do NOT generate entities or relations not present in the path;
            6. Do NOT generate questions based purely on visual attributes (e.g., color, shape, position);
            7. Questions must be natural and clear; answers must be concise and accurate;
            8. **If no image or visual triples are provided, you MUST return an empty result.**

            Output Requirements:
            - Return JSON ONLY
            - No explanation text
            - JSON structure must strictly match the required format
            - If no visual information exists, return:
              {"QA_pairs": []}
            """)

    def build_prompt(self, img_id: str, path: str, vis_triple: List[str]) -> str:

        vis_text = "\n".join([t for t in vis_triple if img_id in t])

        if self.lang == "zh":
            return textwrap.dedent(f"""
            图像ID: {img_id}

            知识图谱路径（Path）：
            {path}

            图像视觉三元组：
            {vis_text if vis_text else "无"}

            任务要求：
            1. 如果没有视觉三元组或图像信息，直接返回：
               {{"QA_pairs": []}}
            2. 如果存在视觉信息：
               - 根据路径中的实体与关系结构生成问题；
               - 问题必须结合图片中的视觉事实；
               - 每个QA必须同时使用：
                 • 路径中的关系逻辑
                 • 图像中的视觉信息；

            强制输出格式（仅返回此JSON）：
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

            Knowledge Graph Path:
            {path}

            Image Visual Triples:
            {vis_text if vis_text else "None"}

            Task Requirements:
            1. If no visual triples or image information exist, directly return:
               {{"QA_pairs": []}}

            2. If visual information exists:
               - Generate questions based on the relational structure of the path;
               - Questions must combine visual evidence from the image;
               - Each QA must use BOTH:
                 • the logical relations in the path
                 • visual facts from the image;

            Mandatory Output Format (return ONLY JSON):
            {{
              "QA_pairs": [
                {{"question": "Question 1", "answer": "Answer 1"}},
                {{"question": "Question 2", "answer": "Answer 2"}}
              ]
            }}
            """)