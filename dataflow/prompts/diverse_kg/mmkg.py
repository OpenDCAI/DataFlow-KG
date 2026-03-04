import textwrap
from dataflow.utils.registry import PROMPT_REGISTRY
from dataflow.core.prompt import PromptABC
import json


@PROMPT_REGISTRY.register()  # type: ignore
class MMKGVisualEntityExtractionPrompt(PromptABC):
    """
    Prompt for extracting visual entities (image analysis) for Multi-Modal Knowledge Graph construction.
    
    Treats each image as a holistic visual entity and extracts:
    - Label: Short identifier representing the main subject
    - Caption: Detailed description of visual content
    - Quality Score: Assessment of image quality and semantic value
    """
    
    def __init__(self, lang: str = "en"):
        self.lang = lang
    
    def build_system_prompt(self) -> str:
        """Build system prompt for visual entity extraction."""
        return textwrap.dedent("""\
            You are an expert in visual content analysis and multi-modal understanding.
            Your task is to analyze images and extract meaningful visual information 
            that can be represented as entities in a knowledge graph.
            
            Guidelines:
            1. **Holistic View**: Treat the entire image as a single visual entity
            2. **Semantic Focus**: Identify the primary subject or message
            3. **Quality Assessment**: Evaluate if the image has clear semantic meaning
            4. **Descriptive Caption**: Provide rich, detailed visual descriptions
            
            Images with low quality (noise, blur, no clear subject) should receive low quality scores.
            
            IMPORTANT: You MUST respond with valid JSON format only. Do not include any markdown formatting, 
            explanations, or additional text. Return only the raw JSON object.
        """)
    
    def build_prompt(self, img_id: str = None) -> str:  # pyright: ignore[reportArgumentType]
        """
        Build user prompt for visual entity extraction.
        
        Args:
            img_id: Optional image identifier for context
            
        Returns:
            Formatted prompt string
        """
        img_context = f" (Image ID: {img_id})" if img_id else ""
        
        return textwrap.dedent(f"""\
            Analyze this image{img_context}. Treat the entire image as a single visual entity.
            
            Provide the following information:
            1. **label**: A short, unique identifier (2-5 words) representing the main subject
               - Examples: "iPhone Product Photo", "Conference Keynote", "Company Logo"
               - Use clear, descriptive language
               
            2. **caption**: A concise, accurate caption describing the visual content
               - Keep it brief: 1-2 sentences maximum (around 100-150 characters)
               - Include only essential information: what is shown, key visual elements
               - Be precise and factual, avoid unnecessary details or verbose descriptions
               - Focus on the main subject and its key characteristics
               
            3. **quality_score**: Image quality and semantic value score (0-10)
               - 0-3: Poor quality (blur, noise, no clear subject, irrelevant)
               - 4-6: Moderate quality (acceptable clarity, some semantic value)
               - 7-10: High quality (clear, informative, strong semantic meaning)
            
            ### Output Format ###
            {{
                "label": "...",
                "caption": "...",
                "quality_score": 0
            }}
            
            Return valid JSON format only.
        """)
