from abc import ABC, abstractmethod
from embeddings.embeddings_visualization import PhoneticEmbsProjection
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from os.path import join
import os

class EmbsExtraction(ABC): 
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def extract_embeddings(self, vocabulary):
        pass

class MatrixFromEmbs(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def similarity_extraction(self, embs_vocabulary):
        pass

class EmbsFastAnalyzer(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def analyze_and_visualize(self):
        pass

# This class extracts phonetic embeddings (matrices) using a phonetic model

class PhoneticEmbsExtraction(EmbsExtraction):
    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model
        self.phoentic_vocabulary = []

    def extract_embeddings(self, vocabulary, phonetic_dict):
        for vocab in vocabulary:
            phon_vocab = self.embeddings_model.embed(vocab, phonetic_dict)
            self.phoentic_vocabulary.append(phon_vocab)
        return self.phoentic_vocabulary

# This class uses the extraced matrices (from the latter) to create a similarity matrix between all the elements.
# Embeddings must reduce with a reduction technique (PCA the only implemented by now)

class PhoneticSimMatrixFromEmbs(MatrixFromEmbs):
    def __init__(self, similarity_metric) -> None:
        super().__init__()
        self.similarity_metric = similarity_metric

    def similarity_extraction(self, vocabulary, embs):
        similarity = self.similarity_metric(embs)
        similarity_dict = {vocabulary[i]: {vocabulary[j]: similarity[i, j] for j in range(len(vocabulary))} for i in range(len(vocabulary))}
        return similarity_dict

class PhoneticEmbsFastAnalyzer(EmbsFastAnalyzer):
    def __init__(self, vocabulary,reduced_embeddings, similarity_metric, path_to_image) -> None:
        self.vocabulary = vocabulary
        self.similarity_metric = similarity_metric
        self.path_to_image = path_to_image
        self.reduced_embeddings = reduced_embeddings
        self.similarity_dict = {}

    def analyze_and_visualize(self):
        # Create an instance of PhoneticSimMatrixFromEmbs and compute similarity
        sim_matrix = PhoneticSimMatrixFromEmbs(self.similarity_metric)
        self.similarity_dict = sim_matrix.similarity_extraction(self.vocabulary, self.reduced_embeddings)

        # Create an instance of PhoneticMapsCreation and visualize the data
        maps_creator = PhoneticEmbsProjection()
        maps_creator.maps_creation(self.vocabulary, self.reduced_embeddings, self.path_to_image)
        
def create_lists_of_embs(w2v, p2v, words):
    # Extract embeddings
    p2v_vectors = []
    w2v_vectors = []

    for word in words:
        print(f"\nWord: {word}")
        p2v_vec = p2v.embed(word)
        w2v_vec = w2v.embed(word)
        print("Phoneme2Vec:", p2v_vec)
        print("Word2Vec:   ", w2v_vec)

        p2v_vectors.append(p2v_vec)
        w2v_vectors.append(w2v_vec)
    return  w2v_vectors, p2v_vectors

def pca_fit_transform_dict_of_embs(p2v_vectors, w2v_vectors):
    # Apply PCA to both sets
    pca = PCA(n_components=2)

    p2v_2d = pca.fit_transform(p2v_vectors)
    w2v_2d = pca.fit_transform(w2v_vectors)  # You may use a different PCA instance if needed

    plotted_dict = {'p2v_2d': p2v_2d, 'w2v_2d': w2v_2d}
    return plotted_dict

def create_plots(words, p2v_2d, w2v_2d, results_folder, model_name, n_prompt):
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot P2V
    axes[0].set_title('Phoneme2Vec PCA (2D)')
    for i, word in enumerate(words):
        axes[0].scatter(p2v_2d[i, 0], p2v_2d[i, 1], color='blue')
        axes[0].annotate(word, (p2v_2d[i, 0], p2v_2d[i, 1]), xytext=(5, 5), textcoords='offset points', color='blue')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].grid(True)

    # Plot W2V
    axes[1].set_title('Word2Vec PCA (2D)')
    for i, word in enumerate(words):
        axes[1].scatter(w2v_2d[i, 0], w2v_2d[i, 1], color='green')
        axes[1].annotate(word, (w2v_2d[i, 0], w2v_2d[i, 1]), xytext=(5, 5), textcoords='offset points', color='green')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].grid(True)

    plt.tight_layout()
    os.makedirs(join(results_folder, model_name), exist_ok=True)
    plt.savefig(join(results_folder, model_name, str(n_prompt)), dpi=300)