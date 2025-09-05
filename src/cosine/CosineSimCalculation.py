from abc import ABC, abstractmethod
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import torch    
import time
import warnings
import numpy as np
import logging

logging.basicConfig(
    filename='log.txt',
    level=logging.ERROR,  # Set to ERROR level to log only NaN occurrences
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Calculator(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def calc():
        pass

class CosineSim(Calculator):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _format_handler(instance):
        # Convert torch.Tensor to numpy array
        if isinstance(instance, torch.Tensor):
            instance = instance.cpu().detach().numpy()
            if len(instance.shape) > 2:
                instance = instance[-1]

        # Convert list to numpy array
        if isinstance(instance, list):
            instance = np.array(instance)

        # Reshape if necessary
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)

        # Handle NaN values
        if np.isnan(instance).any():
            
            # Log the occurrence of NaN values
            logging.error("NaN values detected in the input array: %s", instance)

            # Replace NaN values (e.g., replace with 0)
            instance = np.nan_to_num(instance, nan=0.0)

        return instance

    # It will calcultates the cosine similarity between A and B
    @staticmethod
    def calc(A, B):
        A, B = CosineSim._format_handler(A), CosineSim._format_handler(B)
        cosine = cosine_similarity(A, B)
        return cosine[0][0]

def clean_dataset(dataset_dict):
    cleaned_dict = {}
    for key, value in dataset_dict.items():
        array = np.array(value)
        if not np.isnan(array).any():
            cleaned_dict[key] = value
        else:
            print(key, 'has nan values')
    return cleaned_dict
        
class CosineAnchorTest(Calculator):
    def __init__(self) -> None:
        super().__init__()

    # trained_embs: dictionary with the format {'word': embeddings}. 
    # tuples: tuples with the following format ('ANCHOR', word_to_compare_1, word_to_compare_2, word_to_compare_n...)
    @staticmethod
    def calc(tuples, trained_embs, cosine_calculator: Calculator):
        all_cosines = []
        list_of_existing_embs = trained_embs.keys()
        for elements in tuples:
            cosine_of_the_five = []
            for n_comparison_element in range(len(elements)):
                # This may be a word or a sentence
                comparison_element = elements[n_comparison_element]
                if comparison_element not in list_of_existing_embs:
                    cosine_sim = {comparison_element: 'absent'}
                    print(comparison_element)
                    warnings.warn("You are missing the embeddings. Are you sure this is not a bug?")
                    time.sleep(5)
                    # If the fivet[0] is absent, it's not useful to continue with this fivet 
                    if n_comparison_element == 0:
                        print(comparison_element)
                        warnings.warn("the anchor and its embedding is missing. Are you sure this is not a bug?")
                        for len_needed_for_format in range(3):
                            cosine_of_the_five.append(cosine_sim)
                        break
                else:
                    anchor = trained_embs[elements[0]]
                    embs_of_comparison_element = trained_embs[comparison_element]
                    if str(embs_of_comparison_element) == 'nan' or str(anchor) == 'nan':
                        print(comparison_element)
                        warnings.warn("You are missing the embeddings. Are you sure this is not a bug?")
                        time.sleep(5)
                        break
                    cosine_sim = {comparison_element: cosine_calculator.calc(anchor, embs_of_comparison_element)}
                cosine_of_the_five.append(cosine_sim)
            all_cosines.append(cosine_of_the_five)
        return all_cosines
    
    @staticmethod
    def _to_pandas(cosines, columns=['anchor', 'homophones', 'homophones_score', 'synonyms', 'synonyms_score', 'phon_dist', 'phon_dist_score', 'graph_dist', 'graph_dist_score']):
        data_list = []

        for element in cosines:
            output_list = [item for dictionary in element for item in dictionary.items()]
            output_list = [element for pair in output_list for element in pair]
            data_list.append(output_list)

        df = pd.DataFrame(data=data_list, columns=columns)
        # Scripted: not very elegant and will need to be changed in the future!
        df = df[(df['homophones_score'] != 'EXCLUDED') & (df['synonyms_score'] != 'EXCLUDED') & (df['phon_dist_score'] != 'EXCLUDED') & (df['graph_dist_score'] != 'EXCLUDED')]

        return df

    