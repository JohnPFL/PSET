from ipa_extraction.IpaExtractor import IpaTranscription
from phonetics.save_and_load import PickleLoader, PickleLoader, PickleSaver
import os
import glob
import argparse

def get_phonetic_transcriptions_from_txt(original_path, transcription_path:str, lang:str, batch_size:int=64):
    sentence_list = IpaTranscription.generate_phonetic_transcriptions(original_path, transcription_path, batch_size, language=lang) 

def save_final_transcription(transcription_path:str, file_name:str='final'):
    # It should load all the pkl in the folder and merge them in a single file
    embeddings = {}
    for file in glob.glob(os.path.join(transcription_path, '*.pkl')):
        embs_dict = PickleLoader.load(file)
        embeddings.update(embs_dict)
    
    merged_path = os.path.join(transcription_path, f'{file_name}.pkl')
    PickleSaver.save(embeddings, merged_path)

def main():
    """
    Parses command-line arguments and generates phonetic transcriptions from a cleaned text file.

    Arguments:
        --txt_clean_path (str): Path to the cleaned text file. Required.
        --transcription_path (str): Path to save the generated phonetic transcriptions.
        --lang (str, optional): Language code for phonetic transcription (default: 'en-us').

    Raises:
        ValueError: If --txt_clean_path is not provided.

    Calls:
        get_phonetic_transcriptions_from_txt: Processes the input file and saves transcriptions.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt_clean_path", default="none", help="Path to the txt_clean file")
    parser.add_argument("--transcription_path", help="Path to save the final path for the transcriptions")
    parser.add_argument("--lang", default="en-us", help="Language for the phonetic transcriptions")
    parser.add_argument("--save", default=False, action='store_true', help="If True, it will save the final transcription in a single file")
    args = parser.parse_args()

    if args.txt_clean_path == 'none':
        raise ValueError('You must provide a path for either the txt_clean file, or the anchors, homophones and synonyms files.')
    get_phonetic_transcriptions_from_txt(args.txt_clean_path, args.transcription_path, lang=args.lang)
    
    if args.save:
        save_final_transcription(args.transcription_path)

if __name__ == '__main__':
    main()