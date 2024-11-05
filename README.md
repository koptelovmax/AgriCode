# AgriCode: a tool for code prediction in agricultural texts
The code and supplementary material to support our submission to System Demonstrations of the 31th International Conference of Computational Linguistics (COLING'25)

## Structure of the repository

### Supplementary material:

- **annex/Agriloop_Delphi_Codesystem.(xlsx,png)** -- the full version of the Ecozept's code system used to annotate the interview responses of Agriloop participants

- **annex/survey_classification.xlsx** -- the complete list of results for the 7-class and 15-class settings, for both paragraph and sentence methods, using the original (non-augmented) data, data augmented by replacing adjectives and adverbs using BERT, data augmented using ChatGPT, and data augmented using Retrieval-Augmented Generation (RAG)

- **annex/prompt_engineering.xlsx** -- decription of the prompts used for querrying in the RAG workflow (for the 15-class setting please refer directly to the code)

### Data preparation:

- **workflow/data_loader_paragraphs.py** -- data set construction for the Paragraph method

- **workflow/data_loader_sentences.py** -- data set construction for the Sentence method

### Data augmentation:

- **workflow/data_augmentation (BERT).ipynb** -- train set augmentation by replacing adjectives and adverbs using BERT

- **workflow/data_augmentation (ChatGPT).ipynb** -- train set augmentation using ChatGPT

- **workflow/data_augmentation (RAG 7-paragraphs).ipynb** -- train set augmentation using RAG for paragraphs and the 7-class setting

- **workflow/data_augmentation (RAG 7-sentences).ipynb** -- train set augmentation using RAG for sentences and the 7-class setting

- **workflow/data_augmentation (RAG 15-paragraphs).ipynb** -- train set augmentation using RAG for paragraphs and the 15-class setting

- **workflow/data_augmentation (RAG 15-sentences).ipynb** -- train set augmentation using RAG for sentences and the 15-class setting

### Evaluation:

- **workflow/segment_classification (7 classes).ipynb** -- model learning and evaluation for the 7-class setting

- **workflow/segment_classification (15 classes).ipynb** -- model learning and evaluation for the 15-class setting

### AgriCode tool:

- **Home.py** -- the homepage of the web application, where the selection between the Paragraph and Sentence methods can be made

- **pages/Paragraphs.py** -- text annotation for 7 and 15 classes using the Paragraph method

- **pages/Sentences.py** -- text annotation for 7 and 15 classes using the Sentence method

## Data availability

### Processed data (to reconstruct data augmentation and evaluation):

- **workflow/data/7-classes** -- original and augmented train and test data for both the Paragraph and Sentence methods in the 7-class setting

- **workflow/data/15-classes** -- original and augmented train and test data for both the Paragraph and Sentence methods in the 15-class setting

## To reconstruct the experiments

### Model learning and evaluation for the 7-class setting:

1) Put the code and the data from the **workflow/** directory to your google drive

2) Fix the data set names in "Open the data" section in *segment_classification (7 classes).ipynb* or *segment_classification (15 classes).ipynb* depending on the setting

3) Run in Google Colab:

`workflow/segment_classification (7 classes).ipynb > Runtime > Run all`

or, depending on the setting:

`workflow/segment_classification (15 classes).ipynb > Runtime > Run all`

## To run the web application on a local machine:

### Installation

There are few Python packages that need to be installed to run the application:

1. Install streamit library for running web application
```sh
pip install streamlit
```
2. Install transformers library for natural language processing (NLP)
```sh
pip install transformers
```

3. Install torch library for deep learning and tensor computations
```sh
pip install torch
```
4. Install nltk library for text processing and NLP

```sh
pip install nltk
```
5. Install sentencepiece library for text tokenization

```sh
pip install sentencepiece
```
6. Install numpy library for numerical computations

```sh
pip install numpy
```

### To run the application

```sh
streamlit run Home.py
```
