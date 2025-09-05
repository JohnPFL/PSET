import pandas as pd
import argparse
from embeddings.data_source import CosineDatasetUtility
from cosine.CosineSimCalculation import CosineSim, CosineAnchorTest
from cosine.ScoreComparator import ScoreComparator, ScoreComparatorExtended
from phonetics.save_and_load import PickleLoader

def _check_output_path(output_path):
    if '.csv' not in output_path:
        raise ValueError('Output path for results must be a CSV file')

def main(dataset_path, embeddings_path, output_path):
    """
    Main function to calculate cosine similarities between dataset items and their embeddings, 
    process the results, and output similarity scores and prevalence statistics.
    Args:
        dataset_path (str): Path to the CSV file containing the dataset.
        embeddings_path (str): Path to the file containing extracted embeddings (pickle format).
        output_path (str): Path to save the output CSV file and prevalence statistics.
    Workflow:
        1. Loads the dataset and embeddings.
        2. Transforms the dataset for cosine similarity computation.
        3. Calculates cosine similarities between dataset items and their embeddings.
        4. Depending on the dataset columns, processes and saves the results:
            - If columns 'phon_dist' and 'graph_dist' are absent:
                - Saves cosine similarities for 'anchor', 'homophones', and 'synonyms' items.
                - Computes and saves prevalence and score differences for 'homophones' and 'synonyms'.
            - If columns 'phon_dist' and 'graph_dist' are present:
                - Saves cosine similarities for 'anchor', 'homophones', 'synonyms', 'phon_dist', and 'graph_dist' items.
                - Handles missing or absent scores for 'phon_dist' and 'graph_dist'.
                - Computes and saves prevalence statistics for 'homophones', 'synonyms', 'phon_dist', and 'graph_dist'.
    Outputs:
        - CSV file with cosine similarity scores.
        - Text file with prevalence statistics and score differences.
        - Additional CSV files with rows containing NaN or absent values (if any). Useful to check if the process went as expected.
    """

    print(f'Calculating cosine similarities... for embeddings:', embeddings_path)
    _check_output_path(output_path)

    pl = PickleLoader()
    
    dataset = pd.read_csv(dataset_path)
    extracted_embs = pl.load(embeddings_path)

    cosine_data_transformer = CosineDatasetUtility()
    transformed_dataset = cosine_data_transformer.apply(dataset)

    cosine_sim_calculator = CosineSim()
    cos_similarities = CosineAnchorTest().calc(transformed_dataset, extracted_embs, cosine_sim_calculator)

    # This is scripted and not very elegant if future expansions will be needed. To be changed in the future!
    if 'phon_dist' not in dataset or 'graph_dist' not in dataset:
        df_cos_similarities = CosineAnchorTest()._to_pandas(cos_similarities, columns=['anchor', 'anchor_score', 'homophones', 'homophones_score',
                                                                                    'synonyms', 'synonyms_score'])
        df_cos_similarities.to_csv(output_path)
        sc = ScoreComparator(df_cos_similarities)
        scores = sc.compare_scores()

        with open(output_path + '_prevalences.txt', 'w') as f: 
            f.write(f'homophones prevalence: {scores[0]}\n')
            f.write(f'synonyms prevalence: {scores[1]}\n')
    else:
        df_cos_similarities = CosineAnchorTest()._to_pandas(cos_similarities, columns=['anchor', 'anchor_score', 'homophones', 'homophones_score',
                                                                                    'synonyms', 'synonyms_score', 'phon_dist', 'phon_dist_score', 'graph_dist', 'graph_dist_score'])
        df_cos_similarities.to_csv(output_path)
        nan_rows = df_cos_similarities[df_cos_similarities.isna().any(axis=1)]
        absent_rows_d = df_cos_similarities[df_cos_similarities['phon_dist_score'] == "absent"]
        absent_rows_e = df_cos_similarities[df_cos_similarities['graph_dist_score'] == "absent"]
        if not nan_rows.empty or not absent_rows_d.empty:
            print(f'Warning: NaN values found in the dataset. Dropping rows with NaN values.')
            nan_rows.to_csv(output_path + '_nan_rows_phon_dist.csv')
            absent_rows_d.to_csv(output_path + '_absent_rows_phon_dist.csv')
            df_cos_similarities = df_cos_similarities.dropna()
            df_cos_similarities = df_cos_similarities[df_cos_similarities['phon_dist_score'] != "absent"]
        if not nan_rows.empty or not absent_rows_e.empty:
            print(f'Warning: NaN values found in the dataset. Dropping rows with NaN values.')
            nan_rows.to_csv(output_path + '_nan_rows_graph_dist.csv')
            absent_rows_d.to_csv(output_path + '_absent_rows_graph_dist.csv')
            df_cos_similarities = df_cos_similarities.dropna()
            df_cos_similarities = df_cos_similarities[df_cos_similarities['graph_dist_score'] != "absent"]            
        sc = ScoreComparatorExtended(df_cos_similarities)
        scores = sc.compare_scores()
        with open(output_path + '_prevalences.txt', 'w') as f:
            f.write(f'homophones prevalence: {scores[0]}\n')
            f.write(f'synonyms prevalence: {scores[1]}\n')
            f.write(f'phon_dist prevalence: {scores[2]}\n')
            f.write(f'graph_dist prevalence {scores[3]}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate cosine similarities for a dataset.')
    parser.add_argument('--dataset_path', type=str, help='Path to the CSV file containing the dataset.')
    parser.add_argument('--embeddings_path', type=str, help='Path to the file containing the extracted embeddings.')
    parser.add_argument('--output_path', type=str, help='Path to save the output CSV file with cosine similarities.')

    args = parser.parse_args()
    main(args.dataset_path, args.embeddings_path, args.output_path)
