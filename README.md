# Purdue Grade Predictor

## What This Is

Every Purdue student faces the same problem at registration: you can see a course name and a professor, but you have almost no signal about what you're walking into. RateMyProfessor has sparse, subjective reviews. BoilerGrades shows raw historical grade distributions, but you have to manually dig through years of data and do the math yourself.

This project automates that signal. Give it a course number, a professor, and a semester and it returns a predicted probability distribution across five grade outcomes: A, B, C, D/F, and withdrawal. Not just "this course averages a 3.2 GPA" but "there's a 61% chance you get an A, 24% chance of a B, 10% chance of a C, 4% chance of D/F, and 2% chance of withdrawal." That's a meaningfully more useful answer for a student deciding between the same course with Professor X versus Professor Y.

## Results

Five XGBoost models (one per grade bucket) trained on 85,053 Purdue course sections, Summer 2016–Fall 2025. Evaluated on a strictly temporal holdout — the test set is Fall 2024, Spring 2025, and Fall 2025, semesters the model never saw during training (`data/data_split.py`).

| Bucket   | Baseline RMSE | Model RMSE | Improvement |
|----------|---------------|------------|-------------|
| A-rate   | 0.2411        | 0.1888     | 22%         |
| B-rate   | 0.1688        | 0.1431     | 15%         |
| C-rate   | 0.0928        | 0.0752     | 19%         |
| D/F-rate | 0.0603        | 0.0471     | 22%         |
| W-rate   | 0.0416        | 0.0421     | ~baseline   |

*(Reproduced by running `src/models/baseline.py` and `src/train.py` against the current `data/processed/features.csv`.)*

Baseline is a naive lookup of each course's own historical bucket rate, with no model behind it. Model RMSE is the XGBoost ensemble evaluated on the temporal test split. Val and test performance are nearly identical, which suggests the model isn't overfitting to recent semesters. Withdrawal rate is essentially unpredictable from historical grade patterns alone — it's driven more by personal circumstances the model has no signal on.

## How It Works

### Data

Raw data comes from two sources, merged in `data/data_loader.py`:

- **Excel files** obtained via Purdue public records requests, covering most semesters from Summer 2016 through Fall 2023 (`Sum16-Sum21`, `Spring 2022`, `Summer 2022`, `Fall 2022`, `Spring 2023`, `Fall 2023`). Three distinct sheet layouts required three separate parsers (`clean1`, `clean2`, `clean3` in `data/data_cleaner.py`).
- **CSVs** pulled directly from the [BoilerGrades](https://github.com/eduxstad/boiler-grades) GitHub repository, which fill in the semesters the Excel export skipped (Fall 2021, Summer 2023) and extend coverage through Fall 2025.

Grade percentages arrive on different scales depending on the source, so `load_all_data()` normalizes every grade column to a 0–1 fraction before the two sources are combined — CSV values already end up as fractions during CSV cleanup, while Excel values are raw percentages that get divided by 100. This is a spot worth double-checking if new source formats are added, since it currently relies on a `<= 1` threshold to decide whether a value has already been converted.

After cleaning: 85,053 section-level rows, each representing one section of one course in one semester taught by one instructor, with the full grade distribution as proportions across 14 letter-grade categories.

### Feature Engineering

Rather than predicting average GPA (which collapses a bimodal distribution and a uniform distribution into the same number), the model predicts five grade-bucket probabilities directly (`calc_grade_buckets` in `data/feature_engineering.py`). For each bucket, three levels of historical rates are computed:

- **Professor-level rate** — this instructor's historical fraction of students in each bucket, across all prior semesters
- **Course-level rate** — same computation across all instructors who have ever taught this course
- **Department-level rate** — same computation across all courses in this subject

Every historical feature uses a strictly leak-free rolling window — `shift().expanding().mean()` ensures each row's features are computed only from semesters strictly before that row's own semester. The train/val/test split is purely chronological (`data/data_split.py`), with no random shuffling.

### Three-Level Bayesian Shrinkage

The key modeling insight is that raw historical rates are unreliable for professors or courses with little history. A professor's first-semester A-rate shouldn't be trusted as much as their tenth-semester average. `calc_bayesian_rate` in `data/feature_engineering.py` handles this with a hierarchical shrinkage:

**Step 1** — shrink the course's own rate toward the department's rate, weighted by how many prior semesters the course has been taught:

```
μ̂_course = (n_c / (n_c + k_c)) · x̄_course + (k_c / (n_c + k_c)) · μ_dept
```

**Step 2** — shrink the professor's own rate toward the already-blended course estimate, weighted by professor experience:

```
μ̂_final = (n_p / (n_p + k_p)) · x̄_prof + (k_p / (n_p + k_p)) · μ̂_course
```

New professors (no prior history) automatically fall back to the blended course estimate; new courses fall back to the department rate. Prior strength is currently fixed at `k_p = k_c = 5` rather than tuned — a reasonable next step would be grid-searching these alongside the XGBoost hyperparameters.

### Modeling

Five independent XGBoost regressors, one per grade bucket, each trained on 25 features: the Bayesian-blended rate, the raw professor/course/department historical rates for all five buckets, semester type (Fall/Spring/Summer), a course-level tier (1: below 30000, 2: 30000–49999, 3: 50000+, roughly introductory/upper-level/graduate), a COVID-semester flag (Spring 2020 through Spring 2021), and professor/course experience counts (`src/models/model.py`).

`n_estimators=400`, `max_depth=6`, `learning_rate=0.05` were selected via grid search over those three parameters (`src/tuning.py`); `subsample` and `colsample_bytree` are fixed at 0.8. At inference time the five outputs would be renormalized to sum to 1 (not yet implemented — see Limitations).

XGBoost was chosen over neural networks because the problem is tabular with engineered features — tree-based models consistently outperform deep learning on structured tabular data of this size, run on CPU without GPU requirements, and produce interpretable feature importances.

## Project Structure

```
boiler-predict/
├── data/
│   ├── data_cleaner.py        # 3 parsers for 3 Excel sheet formats
│   ├── data_loader.py         # merges Excel + BoilerGrades CSVs, normalizes grade scales
│   ├── feature_engineering.py # rolling historical rates, Bayesian shrinkage, bucket labels
│   ├── data_split.py          # temporal train/val/test split
│   ├── create_csv.py          # runs the full pipeline, writes features.csv
│   ├── raw/grades.xlsx        # gitignored — Excel source
│   └── processed/features.csv # gitignored — engineered feature table
├── src/
│   ├── models/
│   │   ├── baseline.py        # naive historical-rate baseline
│   │   └── model.py           # feature/target lists, train + evaluate functions
│   ├── train.py                # trains and evaluates the 5 XGBoost models
│   ├── tuning.py               # grid search over XGBoost hyperparameters
│   └── eval.py                 # currently empty — placeholder for a standalone eval CLI
├── api/                         # empty — planned FastAPI service, see Future Work
├── ntbks/                       # empty — planned notebooks
├── results/                     # empty — planned place to persist run metrics
└── README.md
```

## Current State vs. Planned Architecture

The pipeline above — data ingestion through trained, evaluated models — is built and working end to end. The serving layer described in earlier planning is **not yet built**: `api/`, `ntbks/`, and `results/` are empty directories, there's no FastAPI app, no database, and no frontend in this repo. `requirements.txt` already includes `fastapi`, `uvicorn`, and `pydantic`, which signals the intended direction:

```
React Frontend (Vercel)
        ↓
FastAPI Backend (Railway)
        ↓
PostgreSQL Database (Railway)   +   Serialized XGBoost Models (joblib)
```

The intent is for the API to load serialized models at startup and serve predictions from a single database lookup for historical context plus in-memory XGBoost inference — no rolling averages recomputed per request, since historical rates would be precomputed offline and stored in the database. None of this is implemented yet, so treat it as a design target, not a shipped feature.

## Key Engineering Decisions

**Why five models instead of one?** Predicting a full probability distribution preserves information that a single GPA scalar throws away — a bimodal weed-out course and a uniformly mediocre course can have the same average GPA but completely different risk profiles. Five separate models let each bucket's predictor specialize on its own signal.

**Why not deep learning?** Deep learning's advantage shows up on unstructured data where the network learns its own representations. Here the representations are already engineered (hierarchical historical rates, shrinkage blends). XGBoost on tabular data with good features tends to beat a neural network on the same data, trains in minutes rather than hours, and produces interpretable feature importances.

**Why temporal split, not random?** A random train/test split would leak future semesters into training — a Fall 2025 row could land in the training set while a Fall 2016 row lands in the test set. That would make val/test RMSE artificially optimistic and produce a model that fails in production, where predictions are always about the future relative to the training data. Every split decision was made with that deployment reality in mind.

## Limitations

- **Withdrawal rate is unpredictable** — personal circumstances (job offers, health, transfers) drive withdrawal decisions more than grade history does. The W-rate model performs at baseline.
- **No enrollment size** — not available in the public records data. Section size likely affects grade distributions (large lectures grade differently than small seminars) but can't be controlled for.
- **Name inconsistencies** — instructor names can vary across years and data sources; this isn't currently deduplicated, which slightly degrades professor-level historical features for affected instructors.
- **Distribution shift** — grade patterns over 9 years mean 2016 distributions are somewhat different from 2025 distributions. The temporal feature set captures some of this but can't fully account for long-term drift.
- **Five independent models, no renormalization yet** — predicting five probabilities independently is already a simplification; the current code doesn't even renormalize the five outputs to sum to 1 at inference time. A proper multinomial or Dirichlet regression would respect the sum-to-1 constraint during training rather than after.
- **Shrinkage priors are fixed, not tuned** — `k_p` and `k_c` are hardcoded at 5 rather than selected via grid search.
- **No serving layer yet** — see Current State vs. Planned Architecture above.

## Running Locally

```bash
git clone https://github.com/adwaiyRanjith/purdue-grade-predictor
cd purdue-grade-predictor
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# you'll need data/raw/grades.xlsx (Purdue public-records Excel export) locally —
# it's gitignored and not included in the repo

# build features (downloads BoilerGrades CSVs, reads the local Excel file)
python data/create_csv.py

# evaluate the naive baseline
python src/models/baseline.py

# train and evaluate the 5 XGBoost models
python src/train.py

# grid search over hyperparameters
python src/tuning.py
```

There's no `uvicorn api.main:app` yet — the API described above hasn't been built.

## Future Work

- Build out `api/` — FastAPI service, PostgreSQL-backed historical lookups, serialized model inference
- Renormalize the five bucket predictions to sum to 1 at inference time
- React frontend for course/professor/semester lookup
- Joint multi-output model (e.g. Dirichlet regression) replacing five independent regressors
- Grid-search the Bayesian shrinkage priors (`k_p`, `k_c`) alongside the XGBoost hyperparameters
- Instructor name deduplication across years/sources
- RateMyProfessor difficulty scores via fuzzy name matching
- Automated semester ingestion pipeline — pull new data as Purdue/BoilerGrades releases it, retrain, redeploy
- Confidence intervals on predictions based on historical variance
- Extend to other Big Ten universities via equivalent public records requests
