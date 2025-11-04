from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from mlflow.entities import Trace
from mlflow.genai import scorer
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    ContextRelevance,
    Faithfulness,
    ResponseRelevancy,
    SemanticSimilarity,
)
from ragas.run_config import RunConfig

from rag.config import AppSettings

##############################################################################################

settings = AppSettings()
run_config = RunConfig(
    timeout=380,
    max_workers=1,
)
llm = ChatOpenAI(
    model_name=settings.LLM_MODEL_NAME,
    base_url=settings.LLM_BASE_URL,
    api_key=settings.LLM_API_KEY,
)
evaluator_llm = LangchainLLMWrapper(llm)
hf_embeddings = HuggingFaceEmbeddings(
    model_name='distiluse-base-multilingual-cased-v1',
    cache_folder=str(Path(settings.CACHE_STORAGE_DIR).resolve().joinpath('embeddings')),
)
embeddings_model = LangchainEmbeddingsWrapper(hf_embeddings)

##############################################################################################

@scorer
def answer_relevancy(
    inputs: Mapping[str, Any],
    outputs: str,
    expectations: Mapping[str, Any],
    trace: Trace,
) -> float:
    try:
        score = evaluate(
            dataset=EvaluationDataset.from_list(
                [
                    {
                        'user_input': inputs['question'],
                        'retrieved_contexts': outputs['documents'],
                        'response': outputs['solution'],
                        'reference': outputs['reference'],
                    },
                ]
            ),
            metrics=[ResponseRelevancy()],
            llm=evaluator_llm,
            run_config=run_config,
            embeddings=embeddings_model,
            show_progress=False,
        )
        return score['answer_relevancy'][0]
    except Exception:
        return 0

##############################################################################################

@scorer
def context_relevance(
    inputs: Mapping[str, Any],
    outputs: str,
    expectations: Mapping[str, Any],
    trace: Trace,
) -> float:
    try:
        score = evaluate(
            dataset=EvaluationDataset.from_list(
                [
                    {
                        'user_input': inputs['question'],
                        'retrieved_contexts': outputs['documents'],
                        'response': outputs['solution'],
                        'reference': outputs['reference'],
                    },
                ]
            ),
            metrics=[ContextRelevance(name='context_relevance')],
            llm=evaluator_llm,
            run_config=run_config,
            embeddings=embeddings_model,
            show_progress=False,
        )
        return score['context_relevance'][0]
    except Exception:
        return 0

##############################################################################################

@scorer
def faithfulness(
    inputs: Mapping[str, Any],
    outputs: str,
    expectations: Mapping[str, Any],
    trace: Trace,
) -> float:
    try:
        score = evaluate(
            dataset=EvaluationDataset.from_list(
                [
                    {
                        'user_input': inputs['question'],
                        'retrieved_contexts': outputs['documents'],
                        'response': outputs['solution'],
                        'reference': outputs['reference'],
                    },
                ]
            ),
            metrics=[Faithfulness()],
            llm=evaluator_llm,
            run_config=run_config,
            embeddings=embeddings_model,
            show_progress=False,
        )
        return score['faithfulness'][0]
    except Exception:
        return 0

##############################################################################################

@scorer
def semantic_similarity(
    inputs: Mapping[str, Any],
    outputs: str,
    expectations: Mapping[str, Any],
    trace: Trace,
) -> float:
    try:
        score = evaluate(
            dataset=EvaluationDataset.from_list(
                [
                    {
                        'user_input': inputs['question'],
                        'retrieved_contexts': outputs['documents'],
                        'response': outputs['solution'],
                        'reference': outputs['reference'],
                    },
                ]
            ),
            metrics=[SemanticSimilarity()],
            llm=evaluator_llm,
            run_config=run_config,
            embeddings=embeddings_model,
            show_progress=False,
        )
        return score['semantic_similarity'][0]
    except Exception:
        return 0

##############################################################################################
