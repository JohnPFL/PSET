from cosine.BatchProcessing import TrainingBatch, BatchEmbsExtractor
from embeddings.data_source import DatasetFactory
from phonetics.save_and_load import BatchConcatenator
from embeddings.embeddings_models import SemanticModelFactory
from phonetics.phon_utility import HighestNumberInFolder
from tqdm import tqdm
import argparse

# Not incredibly elegant: to be changed in the future
def string_to_bool(input_str):
    lower_str = input_str.lower()
    if lower_str == "true":
        return True
    elif lower_str == "false":
        return False
    else:
        raise ValueError(f"Invalid input: {input_str}")

class EmbeddingsProcessor:
    def __init__(self, args):
        self._load_configuration(args)
        self._initialize_components()

    def _load_configuration(self, args):  
        self.dataset_path = args.dataset_path
        self.dataset_config_selector = DatasetFactory(self.dataset_path)
        self.dataset = self.dataset_config_selector.create_dataset()
        self.training_set, self.length = self.dataset.dataset, len(self.dataset.dataset)
        self.embeddings_path = args.embeddings_path
        self.model = SemanticModelFactory.create_model(args.model)
        self.load_last_batch_bool = string_to_bool(args.load_last_batch)
        self.last_batch = self._calculate_last_batch() if self.load_last_batch_bool else 0
        self.batch_size = int(args.batch_size)

    def _initialize_components(self):
        self.training_batch = TrainingBatch()
        self.batch_processor = BatchEmbsExtractor(self.model, self.embeddings_path, self.batch_size)
        self.last_batch = self._calculate_last_batch() if self.load_last_batch_bool else 0
        return self.training_batch
    
    def _calculate_last_batch(self):
        batch_calculator = HighestNumberInFolder() 
        last_batch = batch_calculator.operate(self.embeddings_path) 
        if last_batch is None: 
            last_batch = 0 
            return last_batch 
        return last_batch - 1 
 
    @staticmethod
    def _word_processing_for_dict(batch):
        return [(word, word) for word in batch]

    # Responsibility: loading the batches in case of existing batches. Extract embeddings 
    def extract_embeddings(self):

        # Actually extracting embeddings 

        for i in tqdm(range(self.last_batch * self.batch_size, self.length, self.batch_size), desc="Processing batches"):
            self.training_batch._get_batch(self.batch_size, i, self.training_set)
            self.training_batch.batch = self._word_processing_for_dict(self.training_batch.batch)
            self.batch_processor._extract_batch_embeddings_as_dict(i, self.batch_size, self.training_batch.batch)

       # Finalizing the batch processing and saving everything at the specified path composed of model name + dataset name
        BatchConcatenator().concatenate_batches(self.embeddings_path + self.model.__class__.__name__
        + '_' + args.dataset_path.split('/')[-1] + '.pkl', self.embeddings_path, self.length // self.batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Embeddings Extraction script')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset file')
    parser.add_argument('--embeddings_path', type=str, required=True, help='Output in which the embeddings will be saved')
    parser.add_argument('--model', deafult='Word2Vec', type=str, required=True, help='Semantic model to use')
    parser.add_argument('--load_last_batch', type=str, required=True, help='Load the last batch (true/false)')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for processing')

    args = parser.parse_args()
    processor = EmbeddingsProcessor(args)
    processor.extract_embeddings()
