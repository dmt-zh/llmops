#!.venv/bin/python3.12

import json
import logging
import shutil
from pathlib import Path
from random import sample

from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar

##############################################################################################

logging_config = {
    'level': logging.INFO,
    'format': '%(asctime)s | %(message)s'
}
logging.basicConfig(**logging_config, datefmt='%Y-%m-%d %H:%M:%S')

##############################################################################################

CWD = Path(__file__).resolve()
CACHE_DIR = CWD.parent
DATASET_DOMAINS = {
    'covidqa': 'medical',
    'pubmedqa': 'medical',
    'cuad': 'legal',
    'finqa': 'finance',
    'tatqa': 'finance',
}
MAIN_DATASET_PATH = CACHE_DIR.joinpath('main_dataset.json')
EVAL_DATASET_PATH = CACHE_DIR.joinpath('eval_dataset.json')

##############################################################################################

def download_and_create_datasets() -> None:
    """Downloading the dataset from Hugging Face and generating data for the application.

    The dataset for evaluating the RAG systems is `galileo-ai/ragbench`.
    A subset of the `test` dataset is used as the main dataset.

    The validation dataset is generated from the main dataset so that the validation data
    contains text responses that will be stored in the vector database.
    """

    logging.info('Started to load data!')
    main_data = {}
    eval_data = []
    acc = 1

    for dataset_name, domain_name in DATASET_DOMAINS.items():
        raw_dataset = load_dataset(
            path='rungalileo/ragbench',
            name=dataset_name,
            split='test',
            cache_dir=CACHE_DIR,
        )
        filtered_data = raw_dataset.filter(lambda example: bool(example['adherence_score']) and len(''.join(example['documents'])) > 2500)
        sample_ids = sample(range(len(filtered_data)), k=75)
        main_samples = filtered_data.select(sample_ids)

        for doc_row in main_samples:
            main_data[f'doc_{acc}'] = doc_row.get('documents')
            acc += 1

        eval_ids = sample(range(len(main_samples)), k=15)
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

    with (
        open(MAIN_DATASET_PATH, 'w', encoding='utf-8') as fout_main,
        open(EVAL_DATASET_PATH, 'w', encoding='utf-8') as fout_eval,
    ):
        json.dump(main_data, fout_main, indent=4, ensure_ascii=False)
        json.dump(eval_data, fout_eval, indent=4, ensure_ascii=False)
        logging.info(f'Loaded and saved main dataset in "{MAIN_DATASET_PATH}"')
        logging.info(f'Loaded and saved evaluation dataset in "{EVAL_DATASET_PATH}"')

    for path_ in CACHE_DIR.glob('*[!json][!py]'):
        if path_.is_dir():
            shutil.rmtree(path_)
        else:
            path_.unlink()
    logging.info('Cleaned cache dirs.')

##############################################################################################

if __name__ == '__main__':
    disable_progress_bar()
    download_and_create_datasets()
