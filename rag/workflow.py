import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

from ddgs import DDGS
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic_settings import BaseSettings

from rag.evaluation import EvaluationChains

##############################################################################################

class State(TypedDict):
    """
    State structure for LangGraph RAG workflow.

    This structure defines all the data that flows through the LangGraph RAG pipeline.
    Each step in the workflow can read from and write to this state, allowing
    for complex decision-making and proper error handling.
    """

    question: str
    solution: str
    reference: str | None
    web_search: bool
    recursion_limit: int
    documents: Sequence[Document]
    solution_evaluation: float | int
    question_evaluation: float | int
    document_evaluations: Sequence[Mapping[str, Any]]

##############################################################################################

class RAGWorkflow:
    """
    Manages the RAG workflow using LangGraph.

    This class orchestrates the RAG pipeline using LangGraph's state
    management system. It handles document processing, question answering,
    and evaluation with proper error handling and fallback mechanisms.
    """

    def __init__(self, settings: BaseSettings, vector_store: QdrantVectorStore) -> None:
        self._chains = EvaluationChains(settings)
        self._retriever = vector_store.get_retriever
        self._settings = settings
        self._graph = self._buid_graph()

    ##########################################################################################

    def _buid_graph(self) -> StateGraph:
        """Create and configure the state graph for handling queries."""

        graph_builder = StateGraph(State)
        graph_builder.add_node('Retrieve Documents', self._retrieve)
        graph_builder.add_node('Evaluate Documents', self._evaluate)
        graph_builder.add_node('Search Online', self._search_online)
        graph_builder.add_node('Generate Answer', self._generate_answer)

        graph_builder.set_entry_point('Retrieve Documents')
        graph_builder.add_edge('Retrieve Documents', 'Evaluate Documents')
        graph_builder.add_conditional_edges(
            'Evaluate Documents',
            self._retrieved_docs_relevant,
            {
                'Search Online': 'Search Online',
                'Generate Answer': 'Generate Answer',
            },
        )
        graph_builder.add_conditional_edges(
            'Generate Answer',
            self._check_solution,
            {
                'Hallucinations detected': 'Generate Answer',
                'Answers Question': END,
                'Question not addressed': 'Search Online',
            },
        )
        graph_builder.add_edge('Search Online', 'Generate Answer')
        return graph_builder.compile()

    ##########################################################################################

    @property
    def graph(self) -> CompiledStateGraph:
        """Accessing the compiled graph."""

        return self._graph

    ##########################################################################################

    def _retrieve(self, state: State) -> Mapping[str, Sequence[str] | str | bool]:
        """Retrieve documents relevant to the user's question."""

        question = state.get('question')

        try:
            documents = self._retriever.invoke(question)
        except Exception as err:
            logging.warning(f'Error retrieving documents: {err}')
            return {'documents': [], 'question': question, 'web_search': True}
        else:
            return {'documents': documents, 'question': question}

    ##########################################################################################

    def _evaluate(self, state: State) -> Mapping[str, Any]:
        """Filter documents based on their relevance to the question."""

        question = state.get('question', [])
        documents = state.get('documents', [])
        filtered_documents = []
        document_evaluations = []
        min_threshold = 0.75
        total_score = 0.7

        for document in documents:
            agent_response = self._chains.evaluate_retrieved_docs.invoke(
                {'question': question, 'document': document.page_content},
            )
            document_evaluations.append(agent_response)
            if agent_response and agent_response.relevance_score >= min_threshold:
                filtered_documents.append(document)

        return {
            'documents': filtered_documents,
            'question': question,
            'web_search': (len(filtered_documents) / len(documents)) < total_score,
            'document_evaluations': document_evaluations,
        }

    ##########################################################################################

    def _retrieved_docs_relevant(self, state: State):
        """Determine whether retrieved documents are relevant, if not trigger online searching."""

        web_search = state.get('web_search', False)
        return 'Search Online' if web_search else 'Generate Answer'

    ##########################################################################################

    def _search_online(self, state: State) -> Mapping[str, Any]:
        """Search online for additional context if needed."""

        question = state['question']
        documents = state['documents']

        search_results = DDGS().text(query=question, safesearch='off', max_results=self._settings.K_SEARCH_RESULTS)
        web_documents = [Document(page_content=source['body'].strip()) for source in search_results]

        if documents is not None:
            for web_doc in web_documents:
                documents.append(web_doc)
        else:
            documents = web_documents

        return {
            'documents': documents,
            'question': question,
        }

    ##########################################################################################

    def _generate_answer(self, state: State) -> Mapping[str, Any]:
        """Generate an answer based on the retrieved documents."""

        question = state['question']
        documents = state['documents']
        reference = state.get('reference')
        recursion_limit = state.get('recursion_limit', 0) + 1

        solution = self._chains.generate_answer.invoke(
            {'context': documents, 'question': question},
        )
        return {
            'question': question,
            'solution': solution,
            'reference': reference,
            'documents': [doc.page_content if isinstance(doc, Document) else doc for doc in documents],
            'recursion_limit': recursion_limit,
        }

    ##########################################################################################

    def _check_solution(self, state: State) -> str:
        """Evaluate generated answers for completeness and hallucinations presence."""

        question = state['question']
        documents = state['documents']
        solution = state['solution']
        min_threshold = 0.65
        nodes_number = 4

        if state.get('recursion_limit', 0) >= max(self._settings.RECURSION_LIMIT - nodes_number, nodes_number):
            return 'Answers Question'

        solution_evaluation = self._chains.evaluate_solution.invoke(
            {'documents': documents, 'solution': solution},
        )
        if solution_evaluation.score and solution_evaluation.relevance_score > min_threshold:
            question_evaluation = self._chains.evaluate_question.invoke(
                {'question': question, 'solution': solution},
            )
            state['solution_evaluation'] = solution_evaluation.relevance_score
            state['question_evaluation'] = question_evaluation.relevance_score

            if question_evaluation.score and question_evaluation.relevance_score > min_threshold:
                return 'Answers Question'
            return 'Question not addressed'

        return 'Hallucinations detected'

    ##########################################################################################

    def draw_gpraph(self) -> None:
        """Drawing graph and saving its image to `static` folder."""

        save_path = Path(__file__).parent.parent.joinpath('static/graph.png')
        mermaid_png = self._graph.get_graph(xray=5).draw_mermaid_png(
            max_retries=5,
            retry_delay=2.0,
            frontmatter_config={'title': 'RAG Workflow'},
        )
        with open(save_path, 'wb') as fout:
            fout.write(mermaid_png)

##############################################################################################
