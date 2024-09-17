import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from collections import Counter
import spacy

"""
Question 1
This question consists of multiple CSV files (In the Zipped Folder) with ‘large texts’ in one of the columns in each file. Your job is to use the open-source NLP (Natural Language Processing) libraries and perform various tasks.
"""

"""
Task 1:
Extract the ‘text’ in all the CSV files and store them into a single ‘.txt file’.

"""


def task_1() -> None:
    file_names = ["CSV1.csv", "CSV2.csv", "CSV3.csv", "CSV4.csv"]
    all_text = []

    for file_name in file_names:
        df = pd.read_csv(file_name)
        if "TEXT" in df.columns:
            all_text.extend(df["TEXT"])
        elif "SHORT_TEXT" in df.columns:
            all_text.extend(df["SHORT_TEXT"])

    with open("combined_text.txt", "w") as file:
        for text in all_text:
            file.write(text + "\n")
    print("Task 1 completed")


"""
Task 2: Research
Install the libraries(SpaCy – scispaCy – ‘en_core_sci_sm’/’en_ner_bc5cdr_md’).
Install the libraries (Transformers (Hugging Face) - and any bio-medical model (BioBert) that can detect drugs, diseases, etc from the text).

"""


def download_spacy_models():
    """Load spaCy models for general biomedical text and disease/drug NER."""
    nlp_sci = spacy.load("en_core_sci_sm")
    nlp_bc5cdr = spacy.load("en_ner_bc5cdr_md")
    return nlp_sci, nlp_bc5cdr


def load_biobert_model():
    """Load BioBERT model for biomedical NER."""
    model_name = "dmis-lab/biobert-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return pipeline("ner", model=model, tokenizer=tokenizer)


def extract_entities_biobert(text):
    """Extract entities using BioBERT."""
    ner_pipeline = load_biobert_model()
    return ner_pipeline(text)


"""
Task 3: Programming and Research
3.1:
Using any in-built library present in Python, count the occurrences of the words in the text (.txt) and give the ‘Top 30’ most common words.
And store the ‘Top 30’ common words and their counts into a CSV file.

"""


def most_common_30_words():
    """Count the occurrences of words in the text and return the top 30 most common words."""
    with open("combined_text.txt", "r") as file:
        text = file.read()
        words = text.split()
        word_counts = Counter(words)
        top_30_words = word_counts.most_common(30)
        # Store the top 30 words and their counts in a CSV file using pandas with columns "Word" and "Count"
        df = pd.DataFrame(top_30_words, columns=["Word", "Count"])
        df.to_csv("top_30_words.csv", index=False)


def task_3_1():
    most_common_30_words()
    print("Task 3.1 completed")


"""
3.2:
Using the ‘Auto Tokenizer’ function in the ‘Transformers’ library, write a ‘function’ to count unique tokens in the text (.txt) and give the ‘Top 30’ words.
Task 4: Named-Entity Recognition (NER)
Extract the ‘diseases’, and ‘drugs’ entities in the ‘.txt file’ separately using ‘en_core_sci_sm’/’en_ner_bc5cdr_md’ and biobert. And compare the differences between the two models (Example: Total entities detected by both of them, what’s the difference, check for most common words, and check the difference.)

"""


if __name__ == "__main__":
    task_1()
    task_3_1()
