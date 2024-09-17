import pandas as pd

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
Task 3: Programming and Research
3.1:
Using any in-built library present in Python, count the occurrences of the words in the text (.txt) and give the ‘Top 30’ most common words.
And store the ‘Top 30’ common words and their counts into a CSV file.
3.2:
Using the ‘Auto Tokenizer’ function in the ‘Transformers’ library, write a ‘function’ to count unique tokens in the text (.txt) and give the ‘Top 30’ words.
Task 4: Named-Entity Recognition (NER)
Extract the ‘diseases’, and ‘drugs’ entities in the ‘.txt file’ separately using ‘en_core_sci_sm’/’en_ner_bc5cdr_md’ and biobert. And compare the differences between the two models (Example: Total entities detected by both of them, what’s the difference, check for most common words, and check the difference.)

"""


if __name__ == "__main__":
    task_1()
