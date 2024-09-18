# CAS-011-Assignement-2

# HIT 137

# Group 103

## Memebers:

### Md Jabed

### Mohammed Musa

## Initial Setup

1. Clone the repository

```
git clone git@github.com:musamoh/CAS-011-Assignement-2.git
```

2. Change directory to the repository

```
cd CAS-011-Assignement-2
```

3. Create a virtual environment

```
python3 -m venv venv or virtualenv venv
```

4. Activate the virtual environment

```
source venv/bin/activate
```

5. Install the requirements

```
pip install -r requirements.txt
```

7. Install spacy models

```
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

```

6. Install pre-commit hooks

```
pre-commit install
```
