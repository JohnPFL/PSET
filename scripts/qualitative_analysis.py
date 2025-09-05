from embeddings.embeddings_models import Phoneme2Vec, Word2Vec
from phonetics.save_and_load import PickleLoader
from embeddings.data_source import CMUdictionary
import numpy as np
import json
import argparse
from embeddings.embeddings_analyzer import (create_lists_of_embs,
                                            pca_fit_transform_dict_of_embs,
                                            create_plots)


def dict_into_tuple(tup, dict):
    dict_list = list(tup)
    dict_list.append(dict)
    return tuple(dict_list)
        
def check_if_0(l: list):
    for element in l:
        if element.size == 0:
            element = np.zeros()

def main():
    """
    Main function for qualitative analysis of phonetic embeddings.

    This function parses command-line arguments to specify paths for the phoneme2Vec model,
    configuration file, and results directory. It loads the required models (Word2Vec and Phoneme2Vec),
    reads prompts from a configuration JSON file, and processes each prompt to extract word embeddings.
    The embeddings are reduced to 2D using PCA for visualization. Finally, the function generates and
    saves plots comparing phonetic and semantic embeddings for each prompt.

    Command-line Arguments:
        --p2v_path (str): Path to the pre-trained phoneme2Vec model (required).
        --config_path (str): Path to the configuration JSON file (default: 'config.json').
        --results_path (str): Path to the folder where results will be saved (default: './results').

    Raises:
        FileNotFoundError: If the specified config file or model file does not exist.
        json.JSONDecodeError: If the config file is not a valid JSON.
    """
    parser = argparse.ArgumentParser(description='Qualitative Analysis of Phonetic Embeddings')
    parser.add_argument('--p2v_path', type=str, help='Path to the phoneme2Vec pre-trained model')
    parser.add_argument('--config_path', type=str, default='config.json', help='Path to the config file')
    parser.add_argument('--results_path', type=str, help='Path to the results folder')
    args = parser.parse_args()

    # Set default for results_path if not provided
    if args.results_path is None:
        results_path = "./results"
    else:
        results_path = args.results_path

    # Load models
    w2v = Word2Vec()
    p2v = Phoneme2Vec(
        p2v_model=PickleLoader.load(args.p2v_path),
        phonetic_dictionary=CMUdictionary().dataset
    )

    # The format for the {"model_name":"Word2Vec", "row": ["cash, cast, money, bash, cache"]...}
    config_path = args.config_path
    results_path = args.results_path

    with open(config_path, "r") as file:
        data = json.load(file)

    all_words = ()
    embs = ()
    all_words_list = []

    for row in data['rows']:
        all_words = tuple(row.split(','))
        all_words = [word.strip(' ') for word in all_words]
        w2v_vectors, p2v_vectors = create_lists_of_embs(w2v, p2v, all_words)
        if len(embs) == 0:
            embs = tuple()
        embs = dict_into_tuple(embs, { "w2v": w2v_vectors, "p2v": p2v_vectors,})
        all_words_list.append(all_words)
        all_words = tuple()

    # fit transform
    pca_transformed_lists_of_embs = []
    for l in embs:
        # It creates a dictionary with "p2v_2d" and "w2v_2d" as keys and values, so the pca versions of the words.
        pca_transformed_lists_of_embs.append(pca_fit_transform_dict_of_embs(l['p2v'], l['w2v']))

    pca_transformed_lists_of_embs = tuple(pca_transformed_lists_of_embs)

    # savefig
    for i, (all_words, plotted_dict) in enumerate(zip(all_words_list, pca_transformed_lists_of_embs)):
        create_plots(words = all_words,
                    p2v_2d=plotted_dict['p2v_2d'],
                    w2v_2d=plotted_dict['w2v_2d'],
                    results_folder=results_path,
                    model_name=data['model_name'],
                    n_prompt=i)


if __name__ == '__main__':
    main()

