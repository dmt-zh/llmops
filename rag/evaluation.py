from collections.abc import Mapping
from pathlib import Path

import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableSequence
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

##############################################################################################

class EvaluationModel(BaseModel):
    """Model for evaluation of the RAG workflow results."""

    score: bool = Field(
        description='Whether the workflow results are valid - true if valid, false if not.',
    )
    relevance_score: float = Field(
        default=0.5,
        description='Relevance score between 0.0 and 1.0 indicating how certain the evaluation is.',
        ge=0.0,
        le=1.0,
    )

##############################################################################################

class EvaluationChains:
    """
    RAG system evaluations using LLM-based evaluations.

    This class creates structured evaluation pipelines for assessing the quality
    of retrieved documents, questions, and individual documents using predefined prompts
    and evaluation schema.
    """

    def __init__(self, settings: BaseSettings) -> None:
        self._model = ChatOpenAI(
            api_key=settings.LLM_API_KEY,
            model_name=settings.LLM_MODEL_NAME,
            base_url=settings.LLM_BASE_URL,
        )
        self._prompts = self._load_all_prompts()

    ##########################################################################################

    @staticmethod
    def _load_all_prompts() -> Mapping[str, Mapping[str, str]]:
        """Loading all evaluation prompts from a JSON configuration file."""

        working_dir = Path(__file__).resolve().parent
        with open(working_dir.joinpath('prompts.yml')) as fin:
            return yaml.safe_load(fin)

    ##########################################################################################

    def _configure_prompt(self, role: str) -> ChatPromptTemplate:
        """Create a chat prompt template for a specific evaluation role."""

        return ChatPromptTemplate.from_messages(
            [
                ('system', self._prompts.get(role).get('system').strip()),
                ('human', self._prompts.get(role).get('human').strip()),
            ],
        )

    ##########################################################################################

    @property
    def evaluate_retrieved_docs(self) -> RunnableSequence:
        """Evaluation of the relevance and quality of retrieved documents in the RAG system."""

        return self._configure_prompt('retrieval_evaluation') | self._model.with_structured_output(EvaluationModel)

    ##########################################################################################

    @property
    def evaluate_solution(self) -> RunnableSequence:
        """Evaluation the quality and relevance of an individual document."""

        return self._configure_prompt('solution_evaluation') | self._model.with_structured_output(EvaluationModel)

    ##########################################################################################

    @property
    def evaluate_question(self) -> RunnableSequence:
        """Evaluation the quality and clarity of user questions."""

        return self._configure_prompt('question_evaluation') | self._model.with_structured_output(EvaluationModel)

    ##########################################################################################

    @property
    def generate_answer(self) -> RunnableSequence:
        """Final response generated for user's question."""

        return self._configure_prompt('answer_generation') | self._model | StrOutputParser()

##############################################################################################
