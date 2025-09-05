# PSET: A Phonetics-Semantics Evaluation Testbed  

The **Phonetics-Semantics Evaluation Testbed (PSET)** is an English-based benchmark designed to evaluate phonetic embeddings.  

PSET is built on the principle that **phonetic embeddings should prioritize phonetics over semantics**. The testbed leverages:  
- **Homophones** → same sound, different meaning  
- **Synonyms** → different sound, similar meaning  

We evaluate three phonetic embedding models (**Articulatory Embeddings, Phoneme2Vec, XPhoneBERT**) and five large language models (**GPT-4o, Gemini 2.5 Flash, Llama 3.1-8B, OLMo-7B, OLMo 2-7B**).  
- **Phoneme2Vec** performs best among phonetic embeddings.  
- **Gemini 2.5 Flash** outperforms the other LLMs.  

This repository contains the **code, data, and experiments** associated with our **EMNLP 2025 paper**.  

---

## 1. Extracting IPA or ARPAbet Transcriptions  

### IPA Extraction  
- Scripts for extracting **IPA** transcriptions are located in [`scripts/extract_ipa`](scripts/extract_ipa):  
  - `extract_ipa.py` → extract IPA from a single `.csv` or `.txt` file.  
  - `extract_ipas.py` → extract IPA from multiple text files.  
- A pre-transcribed version of the dataset in IPA is already available at:  
  [`data/PSET/IPA/PSET.csv`](data/PSET/IPA/PSET.csv).  

### ARPAbet Extraction  
- For **ARPAbet** transcriptions, we used code from [johnpaulbin’s repository](https://github.com/johnpaulbin), also available on [Kaggle](https://www.kaggle.com/datasets/coldfir4/arpabet?resource=download).  
- We extended this resource by manually integrating missing words.  
- The complete, verified ARPAbet transcriptions are available at:  
  [`data/PSET/ARPA/PSET.txt`](data/PSET/ARPA/PSET.txt).  
- A few words were manually transcribed by us when they were not found in the CMU dictionary.  

---

## 2. Extracting Embeddings  

### Static Embeddings  
1. **`extract_w2v.py`** – extract Word2Vec embeddings.  
   - Run the script by providing a path to the dataset.  
2. **`extract_phonetic_embeddings.py`** – extract phonetic embeddings (setup depends on the model):  
   - **Articulatory embeddings**:  
     - `file_path` → original dataset  
     - `phonetic_path` → IPA-transcribed dataset  
   - **Phoneme2Vec**:  
     - `file_path` → `.txt` file with ARPAbet transcriptions  
     - `p2v_model` → pre-trained Phoneme2Vec model  

### Contextual Embeddings  
- For contextual embeddings, follow the detailed guide in:  
  [`scripts/extract_embeddings/contextual/extract_contextual_embs.md`](scripts/extract_embeddings/contextual/extract_contextual_embs.md)  


---
### 3. PSET: using the dataset 

- `run_pset_test` → To extract the final results, simply run [`scripts/cosine_test/run_pset_test.py`](scripts/cosine_test/run_pset_test.py). Follow the documentation inside of the file.

---

### 4. Using the dataset with LLMs  

1. **`prompting.py`** – example script demonstrating how we tested LLMs and the prompts used.  
   - Currently parameterized for **OLMo**.  
   - Can be easily adapted for other models to reproduce results.  
2. **`extract_results.py`** – script for consistently extracting and formatting LLM results.  

---
