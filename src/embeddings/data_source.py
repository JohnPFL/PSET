from abc import ABC, abstractmethod
import nltk
import pandas as pd
from phonetics.save_and_load import PickleLoader
from nltk.corpus import cmudict

class DatasetUtility(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def apply():
        pass

class DFInterface(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def create_dataset():
        pass

class dataset(ABC):

    def __init__(self) -> None:
        super().__init__()
        self.dataset = []

class CosineDatasetUtility(DatasetUtility):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def apply(dataset):
        tuples_of_words = []
        for l in range(len(dataset)):
            instance = dataset.iloc[l]
            instance = tuple(instance)
            tuples_of_words.append(instance)       
        return tuples_of_words

class CMUdictionary(dataset):
    def __init__(self):
        super().__init__()
        nltk.download('cmudict')
        self.dataset = cmudict.dict()

    def __repr__(self) -> str:
        repr = f'CMU Dictionary. Elements in the dataset: {len(self.dataset)}'
        return repr

    def __getitem__(self, index):
        return self.dataset[index]

class CMUdictionary2Vec(CMUdictionary):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = CMUdictionary2Vec.preparing_for_p2v(self.dataset)

    # this static function should probably be out of this particular class. 
    # this handles the CMU dictionary particularities when extracting phonemese.
    # cases (made-up examples, just to explain):
    # like a -> [[ae][ei]]; hello -> [['h','e','l','o']; ['h','e','l','ou']], 'hi' -> [[ai]]
    @staticmethod
    def list_management_for_cmu(phonemes_list):
        if len(phonemes_list)>1:
            for phoneme in phonemes_list: 
                if len(phoneme) > 1: 
                    phonemes_list = phonemes_list[0]
                    return phonemes_list
                else:
                    phonemes_list = [phon[0] for phon in phonemes_list]
                    return phonemes_list
        else: 
            phonemes_list = phonemes_list[0]
        return phonemes_list
    
    def preparing_for_p2v(self):
            phoneme_sequences = []
            for word, phonemes_list in self.items():
                # Use the first pronunciation variant for simplicity
                phonemes_list = CMUdictionary2Vec.list_management_for_cmu(phonemes_list)
                phoneme_sequences.append(phonemes_list)
            phonetic_dataset = phoneme_sequences
            return phonetic_dataset

# Main way to create the dataset from the csv
class TextToPhoneticDataset(dataset):
    def __init__(self, dataset_path, phonetic_path):
        self.dataset_path = dataset_path
        self.secondary_file_path = phonetic_path
        # Load data based on file extension
        if ".pkl" in self.dataset_path:
            self.dataset = PickleLoader().load(dataset_path)
        elif '.txt' in self.dataset_path:
            self.dataset = self.load_data_txt()
        elif '.csv' in self.dataset_path:
            self.dataset = self.load_data_csv()
        else:
            raise ValueError(f"Format {self.dataset_path.split('.')[1]} not supported by PrePhonDataset class.")

    def load_data_csv(self):
        """ 
        Loads data from a csv file.

        Returns:
            dict: Dictionary containing the loaded data.
        """

        # This is the simple case 
        dataset = pd.read_csv(self.dataset_path)
        dataset = list(dataset.values.flatten())

        if self.secondary_file_path != '':
            secondary_dataset = pd.read_csv(self.secondary_file_path)
            secondary_dataset = list(secondary_dataset.values.flatten())
            dataset = [{a:b} for a, b in zip(dataset, secondary_dataset)]

            # Initialize an empty list to store the flattened content
            flattened_dict = {}

            # Iterate over each dictionary in the list
            for entry in dataset:
                # Concatenate keys and values into the flattened list
                flattened_dict.update(entry)

            dataset = flattened_dict

        return dataset

    def load_data_txt(self):
        """
        Loads data from a text file.

        Returns:
            dict: Dictionary containing the loaded data.
        """
        with open(self.dataset_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            dataset = {}  # Initialize empty dictionary

            for line in lines:
                parts = line.strip().split('|')

                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    dataset[key] = value  # Add data to the dictionary
                
                if len(parts) == 1:
                    key = parts[0].strip()
                    dataset[key] = key

        return dataset

    def get_data(self):
        """
        Returns the loaded data.

        Returns:
            dict: The loaded data.
        """
        return self.dataset

class DatasetFactory(DFInterface):
    def __init__(self, dataset_path, phonetic_path='') -> None:
        self.dataset_path = dataset_path
        self.phonetic_path = phonetic_path

    def create_dataset(self):
        return TextToPhoneticDataset(self.dataset_path, self.phonetic_path)
