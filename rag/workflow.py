import logging
import uuid

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ddgs import DDGS
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic_settings import BaseSettings
from typing import Any, TypedDict
from .evaluation import EvaluationChains


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
    online_search: bool
    documents: Sequence[str]
    document_evaluations: Sequence[Mapping[str, Any]]
    document_relevance_score: Sequence[Mapping[str, Any]]
    question_relevance_score: Sequence[Mapping[str, Any]]

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
        self._k_search_results = settings.K_SEARCH_RESULTS
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
            self._check_hallucinations,
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

        # print("GRAPH STATE: Retrieve Documents") #FIXME:
        question = state.get('question')

        try:
            documents = self._retriever.invoke(question)
            return {'documents': documents, 'question': question}
        except Exception as err:
            logging.warning(f'Error retrieving documents: {err}')
            return {'documents': [], 'question': question, 'online_search': True}

    ##########################################################################################

    def _evaluate(self, state: State) -> Mapping[str, Any]:
        """Filter documents based on their relevance to the question."""

        print("GRAPH STATE: Grade Documents")
        question = state.get('question', [])
        documents = state.get('documents', [])
        filtered_documents = []
        document_evaluations = []

        for document in documents:
            agent_response = self._chains.evaluate_retrieved_docs.invoke(
                {'question': question, 'document': document.page_content},
            )
            document_evaluations.append(agent_response)
            if agent_response and agent_response.relevance_score >= 0.75:
                filtered_documents.append(document)

        return {
            'documents': filtered_documents,
            'question': question,
            'online_search': (len(filtered_documents) / len(documents)) < 0.7,
            'document_evaluations': document_evaluations,
        }

    ##########################################################################################

    def _search_online(self, state: State) -> Mapping[str, Any]:
        """Search online for additional context if needed."""

        # print("GRAPH STATE: Search Online")  #FIXME:
        question = state['question']
        documents = state['documents']

        search_results = DDGS().text(query=question, safesearch='off', max_results=self._k_search_results)
        content = '\n\n'.join([element['body'].strip() for element in search_results])
        results = Document(page_content=content)

        if documents is not None:
            documents.append(results)
        else:
            documents = [results]

        return {
            'documents': documents,
            'question': question,
        }

    ##########################################################################################

    def _retrieved_docs_relevant(self, state: State):
        """Determine whether retrieved documents are relevant, if not trigger online searching."""

        online_search = state.get('online_search', False)
        return 'Search Online' if online_search else 'Generate Answer'

    ##########################################################################################

    def _check_hallucinations(self, state: State) -> str:
        """Check for hallucinations in the generated answers."""

        # print("GRAPH STATE: Check Hallucinations") #FIXME:
        question = state['question']
        documents = state['documents']
        solution = state['solution']

        # print("Checking document relevance...") #FIXME:
        document_relevance_score = self._chains.evaluate_document.invoke(
            {'documents': documents, 'solution': solution},
        )
        if document_relevance_score.score:
            # print("Checking question relevance...") #FIXME:
            question_relevance_score = self._chains.evaluate_question.invoke(
                {'question': question, 'solution': solution},
            )
            state['document_relevance_score'] = document_relevance_score
            state['question_relevance_score'] = question_relevance_score

            if question_relevance_score.score:
                # print("ROUTING DECISION: Going to 'END' (Answers Question)") #FIXME:
                return 'Answers Question'
            # print("ROUTING DECISION: Going to 'Search Online' (Question not addressed)") #FIXME:
            return 'Question not addressed'

        state['document_relevance_score'] = document_relevance_score
        return 'Hallucinations detected'

    ##########################################################################################

    def _generate_answer(self, state: State) -> Mapping[str, Any]:
        """Generate an answer based on the retrieved documents."""

        # print("GRAPH STATE: Generate Answer") #FIXME:
        question = state['question']
        documents = state['documents']

        solution = self._chains.solution_generator.invoke(
            {'context': documents, 'question': question},
        )
        return {'documents': documents, 'question': question, 'solution': solution}

    ##########################################################################################

    def draw_gpraph(self) -> None:
        """Drawing graph and saving its image to `static` folder."""

        save_path = Path(__file__).parent.parent.joinpath('static/graph.png')
        mermaid_png = self._graph.get_graph().draw_mermaid_png(
            frontmatter_config={'title': 'RAG Workflow'}
        )
        with open(save_path,'wb') as fout:
            fout.write(mermaid_png)

##############################################################################################
