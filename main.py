#!.venv/bin/python3.12

import logging
import sys

import click
import mlflow

from config import AppSettings
from data.download_data import create_datasets
from rag import QdrantStorage, RAGWorkflow

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
@click.option('-q', default=None, help='Question to process.')
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

##############################################################################################

main.add_command(create_datasets)
main.add_command(create_collection)
main.add_command(delete_collection)
main.add_command(process_question)

##############################################################################################

if __name__ == '__main__':
    main()
