from typing import final
import pandas as pd
import subprocess
import os
import glob
import argparse

"""
This script extracts IPA (International Phonetic Alphabet) transcriptions from specified columns in one or more CSV files.
It processes each column individually, generates IPA transcriptions using an external script, and outputs new CSV files
with the IPA transcriptions in a specified column order. Temporary files are managed and cleaned up automatically.

Usage:
    python extract_ipas.py --clean_dataset_paths file1.csv file2.csv ... --output_path /path/to/output --columns_order col1 col2 col3

Arguments:
    --clean_dataset_paths : List of paths to input CSV files.
"""

# This function will remove all temporary files except the final ones.
def remove_tmp(if_files):
    for file in if_files:
        # Remove all temporary files except the final ones.
        if '_final_IPA' not in file:
            os.remove(file)

def extract_ipas(paths, definitive_output_path, columns_order):
    """
    Extracts IPA (International Phonetic Alphabet) transcriptions for specified columns in multiple CSV datasets.

    For each CSV file in `paths`, this function:
    - Reads the dataset.
    - For each column, creates a temporary CSV file containing only that column's data.
    - Runs an external script (`extract_ipa.py`) to generate IPA transcriptions for each column.
    - Collects the IPA transcriptions, reconstructs a DataFrame with the same columns as the original (in the specified `columns_order`), and saves the result as a new CSV file in `definitive_output_path`.
    - Cleans up temporary files after processing.

    Args:
        paths (list of str): List of file paths to the input CSV datasets.
        definitive_output_path (str): Directory path where the final IPA CSV files will be saved.
        columns_order (list of str): List specifying the order of columns in the output CSV files.

    Returns:
        list of str: List of file paths to the generated IPA CSV files.
    """
    lists_of_IPA_files = []
    # We extract IPA for each column in each file
    for p in paths: 
        # p must always be a csv file that represents the dataset
        # We create a temporary dataset from the current file:
        tmp_df = pd.read_csv(p)
        # We create a temporary directory to store intermediate files
        os.makedirs('tmp', exist_ok=True)
        # We remove any existing temporary files
        if_files = glob.glob('tmp/*')
        remove_tmp(if_files)

        # We extract IPA for each column in each file: the original extraction works better for single columns
        for column in tmp_df:
            tmp_path = f'tmp/tmp_{str(column)}'
            # We create a temporary dataset for the current column
            with open(f'{tmp_path}.csv', 'w') as f:
                f.write("\n".join(tmp_df[column].astype(str))) 

            # It will run the extract_ipa (that works for a single file) for each one of the paths
            # In our dataset we only have English
            process = subprocess.run(['python', './scripts/extract_ipa.py', 
                                        '--txt_clean_path', f'{tmp_path}.csv', 
                                        '--transcription_path', f'{tmp_path}_IPA.txt',
                                        '--lang', 'en-us'],
                                         stdout=subprocess.PIPE)


        # We create the final dataframe (IPA translated)
        final_dataframe = pd.DataFrame()
        # We find the tmp folder we created before
        list_of_tmp_files = os.listdir('tmp')
        # We join single columns that we extracted before
        for column in list_of_tmp_files:
            if 'IPA' in column:
                df = pd.read_csv(os.path.join('tmp', column), header=None)
                # We need to split the column in two parts, the first one is the original word, the second one is the IPA transcription
                for each_transcription in df.iterrows():
                    df.iloc[each_transcription[0]] = each_transcription[1][0].split('|')[1]
                # We add the column to the final dataframe
                final_dataframe[column.split('_')[1].split('_')[0]] = df[0]

        # We need to reorder the columns
        final_dataframe = final_dataframe.reindex(columns=columns_order)
        # We save the final dataframe
        final_dataframe_path = f'{definitive_output_path}/_{p.split("/")[-1].split(".csv")[0]}_final_IPA.csv'               
        final_dataframe.to_csv(final_dataframe_path, index=False)
        lists_of_IPA_files.append(final_dataframe_path)

    remove_tmp(if_files)
    return lists_of_IPA_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process phonetic transcriptions.')
    parser.add_argument('--clean_dataset_paths', nargs='+', help='List of paths to CSV files', required=True)
    parser.add_argument('--output_path', type=str, required=True, help='Path to the FOLDER to save the definitive IPA transcriptions')
    parser.add_argument('--columns_order', nargs='+', default=['a', 'b', 'c'], help='List of columns in the order they appear in the CSV files')
    args = parser.parse_args()
    
    extract_ipas(args.clean_dataset_paths, args.output_path, args.columns_order)
