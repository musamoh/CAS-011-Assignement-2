import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from collections import Counter
import spacy
import torch
import csv

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


def task_4():
    # Due to the limitation of SpaCy, we will process only the first 1M characters
    text_file = "combined_text.txt"

    # Load spaCy models
    nlp_sci, nlp_bc5cdr = download_spacy_models()
    spacy_entities = Counter()

    with open(text_file, "r") as file:
        text = file.read()
        doc = nlp_sci(
            text[:1000000]
        )  # process only first 1M characters due to SpaCy's limitation
        for ent in doc.ents:
            spacy_entities.update([(ent.text, ent.label_)])
    # print(spacy_entities)

    nlp_bc5cdr_entities = Counter()

    with open(text_file, "r") as file:
        text = file.read()
        doc = nlp_bc5cdr(text[:1000000])
        for ent in doc.ents:
            nlp_bc5cdr_entities.update([(ent.text, ent.label_)])
    # print(nlp_bc5cdr_entities)

    biobert_entities = Counter()
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    model = AutoModelForTokenClassification.from_pretrained(
        "dmis-lab/biobert-base-cased-v1.1"
    )
    labels = ["O", "B-DRUG", "I-DRUG", "B-DISEASE", "I-DISEASE"]
    with open(text_file, "r") as file:
        text = file.read(1000000)
        inputs = tokenizer(
            text, return_tensors="pt", max_length=512, truncation=True, padding=True
        )
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        for token, prediction in zip(inputs["input_ids"][0], predictions[0]):
            if labels[prediction] in ["B-DRUG", "I-DRUG", "B-DISEASE", "I-DISEASE"]:
                biobert_entities.update([tokenizer.decode([token])])

    print(biobert_entities)

    # save to csv
    with open("task_4_spacy_entities.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(["Entity", "Label", "Count"])
        for entity, count in spacy_entities.items():
            writer.writerow([entity, count])

    with open("task_4_biobert_entities.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(["Entity", "Count"])
        for entity, count in biobert_entities.items():
            writer.writerow([entity, count])

    with open("task_4_nlp_bc5cdr_entities.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(["Entity", "Label", "Count"])
        for entity, count in nlp_bc5cdr_entities.items():
            writer.writerow([entity, count])

    # Compare the results
    print(f"Total entities detected by SpaCy:\n {len(spacy_entities)} \n\n")
    print(f"Total entities detected by BioBERT:\n {len(biobert_entities)}\n\n")
    print(f"Common entities:\n {spacy_entities & biobert_entities}\n\n")
    print(f"Entities only detected by SpaCy:\n {spacy_entities - biobert_entities}\n\n")
    print(
        f"Entities only detected by BioBERT:\n, {biobert_entities - spacy_entities}\n\n"
    )
    print("Task 4 completed")


if __name__ == "__main__":
    # task_1()
    # task_3_1()
    # task_3_2()

    task_4()
