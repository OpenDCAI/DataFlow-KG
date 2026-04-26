import pandas as pd
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC


@OPERATOR_REGISTRY.register()
class SchoKGGetOntology(OperatorABC):

    def __init__(self):
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "en") -> tuple:
        if lang == "zh":
            return (
                "SchoKGGetOntology 用于加载学者知识图谱基础本体。",
                "包含实体类型、关系类型、属性类型",
                "输出: entity_type, relation_type, attribute_type"
            )
        return (
            "SchoKGGetOntology loads the basic ontology for SchoKG.",
            "Includes entity types, relation types, attribute types.",
            "Output: entity_type, relation_type, attribute_type"
        )

    # =========================
    # Entity Ontology
    # =========================

    def load_entity_types(self):

        entity_types = {
            "Person":       ["Author", "Reviewer", "Researcher"],
            "Organization": ["University", "ResearchInstitute", "Company", "Publisher", "Funder"],
            "Work":         ["Paper", "Preprint", "ReviewArticle", "DatasetPaper"],
            "Venue":        ["Journal", "Conference", "Workshop", "Source"],
            "Topic":        ["ResearchTopic", "Keyword", "Discipline"]
        }

        return entity_types

    # =========================
    # Relation Ontology
    # =========================

    def load_relation_types(self):

        relation_types = {
            "AuthorshipRelation":   ["author_of"],
            "AffiliationRelation":  ["affiliated_with", "member_of", "employed_by"],
            "PublicationRelation":  ["published_in", "published_by", "accepted_by"],
            "CitationRelation":     ["cites", "is_cited_by"],
            "TopicRelation":        ["has_topic", "related_to_topic"],
            "FundingRelation":      ["funded_by"]
        }

        return relation_types


    def load_attribute_types(self):
        return {
            "PersonAttribute":       ["homepage", "h_index", "orcid"],
            "WorkAttribute":         ["title", "abstract", "year", "doi", "citation_count"],
            "VenueAttribute":        ["venue_name", "issn", "ccf_rank"],
            "OrganizationAttribute": ["organization_name", "country"],
            "TopicAttribute":        ["topic_name"]
        }

    # =========================
    # Run
    # =========================

    def run(self, 
        storage: DataFlowStorage = None
        ):

        self.logger.info("Loading SchoKG ontology")

        entity_types = self.load_entity_types()
        relation_types = self.load_relation_types()
        attribute_types = self.load_attribute_types()

        dataframe = pd.DataFrame({
            "entity_type": [entity_types],
            "relation_type": [relation_types],
            "attribute_type": [attribute_types]
        })

        output_file = storage.write(dataframe, file_path="./.cache/schokg/ontology.json", use_current_step=False)

        self.logger.info(f"Ontology saved to {output_file}")

        return ["entity_type","relation_type","attribute_type"]
