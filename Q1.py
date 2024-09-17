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
        df.to_csv("task_3_1_top_30_words.csv", index=False)


def task_3_1():
    most_common_30_words()
    print("Task 3.1 completed")


"""
3.2:
Using the ‘Auto Tokenizer’ function in the ‘Transformers’ library, write a ‘function’ to count unique tokens in the text (.txt) and give the ‘Top 30’ words.

"""


def count_unique_tokens(file_path: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased", clean_up_tokenization_spaces=True, use_fast=True
    )

    token_counts = Counter()
    chunk_size = 512  # Adjust this value based on your system's memory

    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            text = f.read(chunk_size)
            if not text:
                break
            tokens = tokenizer.tokenize(text)
            token_counts.update(tokens)

    top_30_tokens = token_counts.most_common(30)

    df = pd.DataFrame(top_30_tokens, columns=["Token", "Count"])
    df.to_csv("task_3_2_top_30_tokens.csv", index=False)


def task_3_2():
    file_path = "combined_text.txt"
    count_unique_tokens(file_path)
    print("Task 3.2 completed")


"""
Task 4: Named-Entity Recognition (NER)
Extract the ‘diseases’, and ‘drugs’ entities in the ‘.txt file’ separately using ‘en_core_sci_sm’/’en_ner_bc5cdr_md’ and biobert. And compare the differences between the two models (Example: Total entities detected by both of them, what’s the difference, check for most common words, and check the difference.)

"""


def extract_entities_spacy(text, nlp):
    """Extract entities using spaCy."""
    # Split text into chunks of 1,000,000 characters each
    chunks = [text[i : i + 1000000] for i in range(0, len(text), 1000000)]

    entities = []
    for chunk in chunks:
        doc = nlp(chunk)
        entities.extend([(ent.text, ent.label_) for ent in doc.ents])

    return entities


def task_4():
    text_file = "combined_text.txt"
    with open(text_file, "r") as file:
        text = file.read()

    nlp_sci, nlp_bc5cdr = download_spacy_models()

    # Extract entities using spaCy models
    entities_sci = extract_entities_spacy(text, nlp_sci)
    entities_bc5cdr = extract_entities_spacy(text, nlp_bc5cdr)

    # Extract entities using BioBERT model
    entities_biobert = extract_entities_biobert(text)

    # Compare the differences between the two models
    common_entities_sci = set([ent[0] for ent in entities_sci])
    common_entities_bc5cdr = set([ent[0] for ent in entities_bc5cdr])
    common_entities_biobert = set([ent["word"] for ent in entities_biobert])

    common_entities = common_entities_sci.intersection(
        common_entities_bc5cdr, common_entities_biobert
    )

    print(f"Total entities detected by spaCy (en_core_sci_sm): {len(entities_sci)}")
    print(
        f"Total entities detected by spaCy (en_ner_bc5cdr_md): {len(entities_bc5cdr)}"
    )
    print(f"Total entities detected by BioBERT: {len(entities_biobert)}")
    print(f"Common entities detected by all models: {len(common_entities)}")

    print("Entities detected by spaCy (en_core_sci_sm):")
    print(entities_sci)

    print("Entities detected by spaCy (en_ner_bc5cdr_md):")
    print(entities_bc5cdr)

    print("Entities detected by BioBERT:")
    print(entities_biobert)


if __name__ == "__main__":
    # task_1()
    # task_3_1()
    # task_3_2()

    task_4()
