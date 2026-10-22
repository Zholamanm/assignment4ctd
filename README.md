# Assignment 4 - Internship decision support system

Ranks five internship offers with Weighted Sum (WSM) and TOPSIS.

The decision table is `data/internships.csv`. It is the case matrix for this task, not a downloaded training set.

## Setup

```bash
cd assignment4
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the ranking in the terminal

```bash
python dss.py
```

Prints the matrix, constraint check, normalized table, WSM scores, TOPSIS scores, a comparison, and saves:

- `output/scores.png`
- `output/heatmap.png`

## Run the weight editor

```bash
streamlit run app.py
```

Open the local URL Streamlit prints. Change a weight; the five weights must still sum to 1.00.
