#!.venv/bin/python3.12

from config import AppSettings
from rag import QdrantStorage, RAGWorkflow

settings = AppSettings()
vectore_store = QdrantStorage(settings=settings)
rag_workflow = RAGWorkflow(settings=settings, vector_store=vectore_store)

result = rag_workflow.graph.invoke(
    input={'question': 'How can sustained immunity be generated?'},
    config={'recursion_limit': settings.RECURSION_LIMIT},
)
print(result, flush=True)
