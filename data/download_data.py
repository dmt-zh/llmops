#!.venv/bin/python3.12

import json
import random
import shutil
from pathlib import Path

import click
from datasets import load_dataset

##############################################################################################

CWD = Path(__file__).resolve()
CACHE_DIR = CWD.parent
DATASET_DOMAINS = {
    'pubmedqa': 'medical',
    'cuad': 'legal',
    'finqa': 'finance',
    'techqa': 'tech',
    'hagrid': 'general',
}
MAIN_DATASET_PATH = CACHE_DIR.joinpath('main_dataset.json')
EVAL_DATASET_PATH = CACHE_DIR.joinpath('eval_dataset.json')

##############################################################################################

@click.command()
@click.pass_context
def create_datasets(ctx: click.Context) -> None:
    """Download and create datasets for RAG systems.

    Downloading the dataset from Hugging Face and generating data for the application.

    The dataset for evaluating the RAG systems is `galileo-ai/ragbench`.
    A subset of the `test` dataset is used as the main dataset.

    The validation dataset is generated from the main dataset so that the validation data
    contains text responses that will be stored in the vector database.
    """

    logger = ctx.obj['logger']
    random.seed(7589)
    main_data = {}
    eval_data = []
    acc = 1

    logger.info('Started to load data!')
    for dataset_name, domain_name in DATASET_DOMAINS.items():
        raw_dataset = load_dataset(
            path='rungalileo/ragbench',
            name=dataset_name,
            split='test',
            cache_dir=CACHE_DIR,
        )
        filtered_data = raw_dataset.filter(lambda example: bool(example['adherence_score']) and len(''.join(example['documents'])) > 2500)
        sample_ids = random.sample(range(len(filtered_data)), k=30)
        main_samples = filtered_data.select(sample_ids)

        for doc_row in main_samples:
            main_data[f'doc_{acc}'] = {
                'body': doc_row.get('documents'),
                'question': doc_row.get('question'),
                'response': doc_row.get('response'),
            }
            acc += 1

        eval_ids = random.sample(range(len(main_samples)), k=5)
        eval_samples = main_samples.select(eval_ids)
        messages = []
        for row in eval_samples:
            messages.append({'role': 'user', 'content': row.get('question')})
            messages.append({'role': 'assistant', 'content': row.get('response')})

        if eval_data:
            existing_domains = {seq.get('domain_id') for seq in eval_data}
            if domain_name in existing_domains:
                for samples in eval_data:
                    if samples.get('domain_id') == domain_name:
                        samples['messages'] = samples['messages'] + messages
                        break
            else:
                eval_data.append({'domain_id': domain_name, 'messages': messages})
        else:
            eval_data.append({'domain_id': domain_name, 'messages': messages})
        logger.info(f'Saved "{len(main_samples)}" examples of "{domain_name}" docs in main data')

    with (
        open(MAIN_DATASET_PATH, 'w', encoding='utf-8') as fout_main,
        open(EVAL_DATASET_PATH, 'w', encoding='utf-8') as fout_eval,
    ):
        json.dump(main_data, fout_main, indent=4, ensure_ascii=False)
        json.dump(eval_data, fout_eval, indent=4, ensure_ascii=False)
        logger.info(f'Loaded and saved main dataset: "{MAIN_DATASET_PATH}"')
        logger.info(f'Loaded and saved evaluation dataset: "{EVAL_DATASET_PATH}"')

    for path_ in CACHE_DIR.glob('*[!json][!py]'):
        if path_.is_dir():
            shutil.rmtree(path_)
        else:
            path_.unlink()
    logger.info('Cleaned cache dirs.')

##############################################################################################
