from embeddings.embeddings_models import XPhoneBERT, ClassicBERT, KeyContextExtractor
from ipa_extraction.IpaExtractor import IpaTranscriptionSentence
from phonetics.save_and_load import PickleLoader, PickleSaver
from tqdm import tqdm
import torch
import os
import glob
import argparse
import time

def create_phoneme_key_for_context(dictionary_w_to_s, transcriptor):
    transcriptions_dict = {}
    copy_of_dict = dictionary_w_to_s.copy()
    list_of_keys = copy_of_dict.keys()
    list_of_keys = ' '.join(list_of_keys)
    list_of_phonemes = transcriptor([list_of_keys])
    list_of_phonemes = list_of_phonemes[0].split('▁')
    list_of_keys = list_of_keys.split(' ')
    for key, phoneme in zip(list_of_keys, list_of_phonemes):
        transcriptions_dict[key] = phoneme
    return transcriptions_dict

def get_embeddings(model, w_to_s: dict, embs_path: str, batch_size: int=3, transcriptor = False, window_size:int=20, windows_reduction_anyway=False):
    """
    Extracts contextual embeddings for words within their sentence contexts using a specified model, 
    optionally applying a phonetic transcriptor, and saves the embeddings in batches.

    Args:
        model: The embedding model with a tokenizer and an `embed_from_sentence` method.
        w_to_s (dict): Dictionary mapping target words (keys) to lists of sentences (values) containing those words.
        embs_path (str): Directory path where the embedding batches will be saved as pickle files.
        batch_size (int, optional): Number of words to process before saving a batch of embeddings. Defaults to 3.
        transcriptor (callable or bool, optional): If provided, a function that transcribes a list of strings (sentences) 
            into their phonetic (IPA) representations. If False, no transcription is applied. Defaults to False.
        window_size (int, optional): Maximum number of tokens to consider for each sentence context. Defaults to 20.
        windows_reduction_anyway (bool, optional): If True, always reduce sentence windows to the specified size, 
            regardless of model's max length. Defaults to False.

    Side Effects:
        Saves batches of embeddings as pickle files in the specified directory.

    Returns:
        None
    """
    # If sentence is longer than the model max length, reduced to n tokens (n is window size)
    key_context_extractor = KeyContextExtractor(model_max_length=512, tokenizer = model.tokenizer, window=window_size, windows_reduction_anyway=windows_reduction_anyway)
    if transcriptor is not False:
        transcriptions_dict = create_phoneme_key_for_context(w_to_s, transcriptor)
    else:
        transcriptions_dict = False
    batch_count = 0
    embs_dict = {}
    for key, item in tqdm(w_to_s.items()):
        key_for_context = key
        batch_count += 1
        embs_list = []
        for s in item:
            # If transcriptor is not False, it should be a function that takes a list of strings and return it transcribed in IPA alphabet
            if transcriptions_dict is not False:
                # key is the central word that we want to extract the context from
                key_for_context = transcriptions_dict[key]
            # s is the sentence in which the key is present. If sentence is longer than the model max length, reduced to n tokens (n is window size)
            s = key_context_extractor.check_sentence(s, key_for_context)
            embeddings = model.embed_from_sentence(s, key_for_context)
            embs_list.append(embeddings)
        embs_dict[key] = embs_list
        if batch_count % batch_size == 0:
            path = os.path.join(embs_path, 'batch_' + str(batch_count) + '.pkl')
            PickleSaver.save(embs_dict, path)
            embs_dict = {}

def delete_batch_files(embs_path:str):
     #delete the other files in the folder
    for file in glob.glob(os.path.join(embs_path, '*.pkl')):
        filename = os.path.basename(file)
        if 'batch' in filename:
            os.remove(file)

def torch_simple_list_mean(single_emb):
    single_emb = torch.mean(single_emb, dim=0)
    return single_emb
        

def classic_bert_average(single_emb):
    """
    Averages BERT embeddings across tokens for a single input.

    Args:
        single_emb (torch.Tensor): The input embedding tensor. Can be of shape:
            - (1, seq_len, hidden_dim): Batch of one sequence.
            - (seq_len, hidden_dim): Single sequence.
            - (hidden_dim,): Single token embedding.

    Returns:
        torch.Tensor: The averaged embedding vector of shape (hidden_dim,).

    Notes:
        - Uses `torch_simple_list_mean` to compute the mean across the sequence dimension.
        - Handles input tensors of varying dimensions (3D, 2D, or 1D).
    """
    new_emb = []
    if len(single_emb.shape) == 3:
        single_emb =  torch_simple_list_mean(single_emb[0])
        new_emb.append(single_emb)
    elif len(single_emb.shape) == 2:
        single_emb = torch_simple_list_mean(single_emb)
        new_emb.append(single_emb)
    else:
        new_emb.append(single_emb)
    return new_emb[0]

def xphone_bert_average(single_emb):
    """
    Averages the input embedding tensor based on its dimensionality.

    Parameters:
        single_emb (torch.Tensor): The input embedding tensor. Can be 1D, 2D, or 3D.

    Returns:
        torch.Tensor: The averaged embedding tensor.

    Notes:
        - If `single_emb` is 3D, it averages over the first element (assumed to be a batch of embeddings).
        - If `single_emb` is 2D, it averages over the entire tensor.
        - If `single_emb` is 1D, it returns the tensor as is.

    Requires:
        torch_simple_list_mean: A function to compute the mean of a tensor or list of tensors.
    """
    new_emb = []
    if len(single_emb.shape) == 3:
        single_emb =  torch_simple_list_mean(single_emb[0])
        new_emb.append(single_emb)
    elif len(single_emb.shape) == 2:
        single_emb = torch_simple_list_mean(single_emb)
        new_emb.append(single_emb)
    else:
        new_emb.append(single_emb)
    return new_emb[0]

def final_tensor_average(embs_dict:dict, model:str):
    """
    Averages contextual embeddings for each key in the input dictionary using the specified model's averaging function.
    Args:
        embs_dict (dict): A dictionary where each key maps to a list of embedding tensors.
        model (str): The model type to use for averaging. Supported values are 'ClassicBERT' and 'XPhoneBERT'.
    Returns:
        dict: A dictionary where each key maps to the mean embedding tensor computed from the processed embeddings.
    Raises:
        ValueError: If the specified model is not supported.
    """
    final_embs = {}
    for key, item in embs_dict.items():
        new_item = []
        for single_emb in item:
            if model == 'ClassicBERT':
                single_emb = classic_bert_average(single_emb)
                new_item.append(single_emb)
            elif model == 'XPhoneBERT':
                single_emb = xphone_bert_average(single_emb)
                new_item.append(single_emb)
            else:
                raise ValueError('Model not supported')
                    
        final_embs[key] = torch.mean(torch.stack(new_item), dim=0)

    return final_embs
        
def save_final_embeddings(embs_path:str, model:str, file_name:str='final'):
    # It loads all the pkl in the folder and merge them in a single file
    embeddings = {}
    for file in glob.glob(os.path.join(embs_path, '*.pkl')):
        embs_dict = PickleLoader.load(file)
        embeddings.update(embs_dict)
        
    embeddings = final_tensor_average(embeddings, model)
    merged_path = os.path.join(embs_path, f'{file_name}.pkl')
    PickleSaver.save(embeddings, merged_path)
    delete_batch_files(embs_path)

def main(args):
        dataset_name = args.dataset_name
        w_to_s = PickleLoader.load(args.pset_path)
        embs_path = args.output_folder

        if args.transcriptor == True:
            transcriptor = IpaTranscriptionSentence().generate_phonetic_transcriptions
        else: 
            transcriptor = False
        
        models = {'ClassicBERT': ClassicBERT(), 
        'XPhoneBERT': XPhoneBERT()}
        model = models[args.model]

        if args.only_save:
            save_final_embeddings(embs_path, args.model, file_name=f'{dataset_name}_{model.__class__.__name__}')
        else:
            get_embeddings(model, w_to_s, embs_path, transcriptor=transcriptor, window_size=args.window_size, windows_reduction_anyway=args.windows_reduction_anyway)
            save_final_embeddings(embs_path, args.model, file_name=f'{dataset_name}_{model.__class__.__name__}')

if __name__ == '__main__':
    # If you want to have more clear the inputs here, you can check the documentation "extract_contextual_embs.md"
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='PSET', help='By default this is PSET.')
    parser.add_argument('--pset_path', type=str, help='Path to the PSET .pkl file')
    parser.add_argument('--transcriptor', type=bool, default=False, help='If True, the words are transcribed in IPA alphabet')
    parser.add_argument('--model', type=str, help='Model to use for extracting embeddings')
    parser.add_argument('--output_folder', type=str, help='Path to the output folder')
    parser.add_argument('--window_size', type=int, default=20, help='Window size')
    parser.add_argument('--only_save', type=bool, default=False, help='If True, only save the final embeddings')
    parser.add_argument('--windows_reduction_anyway', type=bool, default=False, help='If True, the sentence is reduced to n tokens (n is window size) anyway')
    args = parser.parse_args()
    main(args)