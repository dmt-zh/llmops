#!.venv/bin/python3.12

import gc
import logging
import sys
from os import environ
from typing import NoReturn

import click
import mlflow
import orjson

from data.download_data import create_datasets
from rag import (
    AppSettings,
    QdrantStorage,
    RAGWorkflow,
    answer_relevancy,
    context_relevance,
    faithfulness,
    semantic_similarity,
)

##############################################################################################

def setup_logger():
    """Create a logger with custom configuration."""

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt='%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger

##############################################################################################

def validate_domain(ctx: click.Context, param: str, value: str) -> str | NoReturn:
    valid_domains = ('medical', 'legal', 'finance', 'tech', 'general')
    if value in valid_domains:
        return value
    raise click.BadParameter(f'Valid domains: [{" ".join(valid_domains)}]')

##############################################################################################

@click.group()
@click.pass_context
def main(ctx: click.Context) -> None:
    """The entrypoint of the project."""

    ctx.ensure_object(dict)
    ctx.obj['config'] = AppSettings()
    ctx.obj['logger'] = setup_logger()

##############################################################################################

@click.command()
@click.pass_context
def create_collection(ctx: click.Context) -> None:
    """Create a new Qdrant collection."""

    QdrantStorage(settings=ctx.obj['config'], logger=ctx.obj['logger']).create_collection()

##############################################################################################

@click.command()
@click.pass_context
def delete_collection(ctx: click.Context) -> None:
    """Delete all data from the Qdrant collection."""

    QdrantStorage(settings=ctx.obj['config'], logger=ctx.obj['logger']).clear_collection()

##############################################################################################

@click.command()
@click.option('--q', default=None, help='Question to process.')
@click.pass_context
def process_question(ctx: click.Context, q: str | None = None) -> None:
    """Run the Q&A processing workflow."""

    mlflow.openai.autolog()
    mlflow.langchain.autolog()
    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment(experiment_name='Tracing Runs')

    settings = ctx.obj['config']
    vectore_store = QdrantStorage(settings=settings, logger=ctx.obj['logger'])
    rag_workflow = RAGWorkflow(settings=settings, vector_store=vectore_store)

    if q is not None:
        result = rag_workflow.graph.invoke(
            input={'question': q},
            config={'recursion_limit': settings.RECURSION_LIMIT},
        )
        click.echo(click.style(f'› {result["question"]}', fg='yellow'))
        click.echo(click.style(f'› {result["solution"]}\n', fg='bright_white'))
        return None

    with open('questions.txt', encoding='utf-8') as fin:
        for question in fin:
                result = rag_workflow.graph.invoke(
                    input={'question': question.strip()},
                    config={'recursion_limit': settings.RECURSION_LIMIT},
                )
                click.echo(click.style(f'› {result["question"]}', fg='bright_yellow'))
                click.echo(click.style(f'› {result["solution"]}\n', fg='bright_white'))
                gc.collect()

##############################################################################################

@click.command()
@click.option(
    '--domain',
    required=True,
    help='One of the domain names: `medical`, `legal`, `finance`, `tech`, `general`',
    callback=validate_domain,
)
@click.pass_context
def rag_evaluation(ctx: click.Context, domain: str) -> None:
    """RAG evaluation on a specific domain."""

    experiment_name = f'{domain.capitalize()} domain'
    mlflow.openai.autolog()
    mlflow.langchain.autolog()
    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment(f'{experiment_name} Traces')
    client = mlflow.MlflowClient()

    settings = ctx.obj['config']
    logger = ctx.obj['logger']
    vectore_store = QdrantStorage(settings=settings, logger=logger)
    rag_workflow = RAGWorkflow(settings=settings, vector_store=vectore_store)

    with open('data/eval_dataset.json') as fin:
        dataset = orjson.loads(fin.read())
        for batch in dataset:
            if batch.get('domain_id') == domain:
                eval_samples = iter(batch.get('messages'))
                break

    logger.info(f'Started RAG evaluation on "{domain}" set!')
    for sample in eval_samples:
        assistant = next(eval_samples)
        rag_workflow.graph.invoke(
            input={'question': sample.get('content'), 'reference': assistant.get('content')},
            config={'recursion_limit': settings.RECURSION_LIMIT},
        )
        gc.collect()

    experiment = client.get_experiment_by_name(f'{experiment_name} Traces')
    traces = mlflow.search_traces(
        locations=[experiment.experiment_id],
        filter_string='status = "OK"',
    )

    environ['MLFLOW_GENAI_EVAL_MAX_WORKERS'] = '1'
    mlflow.set_experiment(f'{experiment_name} Evaluation')
    mlflow.genai.evaluate(
        data=traces,
        scorers=[
            answer_relevancy,
            context_relevance,
            faithfulness,
            semantic_similarity
        ],
    )
    logger.info('RAG evaluation is completed!')
    # ./main.py rag-evaluation --domain medical

##############################################################################################

main.add_command(create_datasets)
main.add_command(create_collection)
main.add_command(delete_collection)
main.add_command(process_question)
main.add_command(rag_evaluation)

##############################################################################################

if __name__ == '__main__':
    main()
