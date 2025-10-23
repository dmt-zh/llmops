#!.venv/bin/python3.12

import mlflow

from config import AppSettings
from rag import QdrantStorage, RAGWorkflow

mlflow.openai.autolog()
mlflow.langchain.autolog()
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment(experiment_name='Tracing Runs')


settings = AppSettings()
vectore_store = QdrantStorage(settings=settings)
rag_workflow = RAGWorkflow(settings=settings, vector_store=vectore_store)


def main():
    result = rag_workflow.graph.invoke(
        input={'question': 'How can present systems of surveillance be used?'},
        config={'recursion_limit': settings.RECURSION_LIMIT},
    )
    print(result['question'], flush=True)
    print(result['solution'], flush=True)
    print('-' * 120)


if __name__ == '__main__':
    main()
