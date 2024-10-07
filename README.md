# AgriCode: A Tool for Code Prediction In Agricultural Texts
The code and supplementary material to support our submission to System Demonstrations of the 31th International Conference of Computational Linguistics (COLING'25)

## Structure of the repository:

### Supplementary material

- **Agriloop_Delphi_Codesystem.(xlsx,png)** -- the full version of the Ecozept's code system used to annotate the interview responses of Agriloop participants

- **survey_classification.xlsx** -- the complete list of results for the 7-class and 15-class settings, for both paragraph and sentence methods, using the original (non-augmented) data, data augmented by replacing adjectives and adverbs using BERT, data augmented using ChatGPT, and data augmented using Retrieval-Augmented Generation (RAG)

- **prompt_engineering.xlsx** -- decription of the prompts used for querrying in the RAG workflow (for 15-class setting please refer directly to the code)

### Installation (to run the web application on a local machine)

There are few Python packages that need to be installed to run the application

1. Install streamit library for running web application:
```sh
pip install streamlit
```
2. Install transformers library for natural language processing (NLP):
```sh
pip install transformers
```

3. Install torch library for deep learning and tensor computations:
```sh
pip install torch
```
4. Install nltk library for text processing and NLP:

```sh
pip install nltk
```
5. Install sentencepiece library for text tokenization:

```sh
pip install sentencepiece
```
6. Install numpy library for numerical computations:

```sh
pip install numpy
```

### To run the web application

```sh
streamlit run Home.py
```

To be wrtten more.
