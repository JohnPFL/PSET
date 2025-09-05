import transformers
import torch
import csv
import random
import json
import time
from tqdm import tqdm

### The prompts ####
# This script is just an example of what we've done, since Olmo, for example, requires different code to be queried.
# Proprietary models have their own APIs. We won't include code for them here.


SIMPLE_PROMPT1 = "Which word is more phonetically similar to [ANCHOR]: [WORD1], [WORD2], [WORD3] or [WORD4]? Only respond with the correct word."
SIMPLE_PROMPT2  = "Which word sounds more like [ANCHOR]: [WORD1], [WORD2], [WORD3] or [WORD4]? Only respond with the correct word."
dataset_path = 'PSET-a-Phonetics-Semantics-Evaluation-Testbed/data/PSET/PSET.csv'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def read_dataset(): 
    """ Read in the dataset"""
    # the anchors are the keys
    dataset = {}
    with open(dataset_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for row in csv_reader:
            print(row)
            dataset[row['anchor']] = {
                'homophones': row['homophones'],
                'synonyms': row['synonyms'],
                'phon_dist': row['phon_dist'],
                'graph_dist': row['graph_dist']
            }
    return dataset

def generate_prompts():
    """ Generate the prompts"""
    random.seed(389)

    # Read in the dataset
    dataset = read_dataset()

    prompts = []
    id = 0

    for anchor, data in dataset.items():
        id = id + 1

        # Randomly switch order
        word1 = data['homophones']
        word2 = data['synonyms']
        word3 = data['phon_dist']
        word4 = data['graph_dist']

        sp1 = (id, SIMPLE_PROMPT1.replace('[ANCHOR]', anchor).replace('[WORD1]', word1).replace('[WORD2]', word2).replace('[WORD3]', word3).replace('[WORD4]', word4), data['homophones'])
        sp2 = (id, SIMPLE_PROMPT1.replace('[ANCHOR]', anchor).replace('[WORD1]', word3).replace('[WORD2]', word4).replace('[WORD3]', word2).replace('[WORD4]', word1), data['homophones'])
        sp3 = (id, SIMPLE_PROMPT2.replace('[ANCHOR]', anchor).replace('[WORD1]', word1).replace('[WORD2]', word2).replace('[WORD3]', word3).replace('[WORD4]', word4), data['homophones'])
        sp4 = (id, SIMPLE_PROMPT1.replace('[ANCHOR]', anchor).replace('[WORD1]', word3).replace('[WORD2]', word4).replace('[WORD3]', word2).replace('[WORD4]', word1), data['homophones'])

        prompts.extend([sp1, sp2, sp3, sp4])

    return prompts

def from_message_to_prompt(message):
    chat = [{"role": "user", "content": message}]
    return chat

def from_answer_to_response(completion):
    return completion[0]['generated_text'][1]['content']


def query_llms(pipeline, messages):

    chat = from_message_to_prompt(messages)
    
    while True:
        outputs = pipeline(
                            chat,
                            max_new_tokens=256,
                        )
        completion = from_answer_to_response(outputs)
        return completion

# This is the default for LLama, of course other LLMs may need other pipelines/code
def run_experiment(model_id:str,):

    pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    )
    

    prompts = generate_prompts()

    result = []

    for promptdata in tqdm(prompts, desc='Prompting'):
        id, prompt, gold_answer = promptdata

        messages = [prompt]
        
        completion = query_llms(pipeline, messages)

        result.append(
            {'promptdata': promptdata,
            'response': completion})
        time.sleep(2)

    with open(f'llama3_results_with_pipeline.json', 'w') as fp:
        json.dump(result, fp, indent=4, sort_keys=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    args = parser.parse_args()
    run_experiment()
    