# Stage-Adaptive Stacked Ensemble for Early Student Performance Prediction

A machine learning system that predicts students' final academic performance using early programming activity across five cumulative assignment stages from the CodeWorkout dataset (Spring 2019 and Fall 2019 releases).

This project investigates whether dynamically learning separate ensemble weights for **K-Nearest Neighbors (KNN)**, **Support Vector Regressor (SVR)**, and **XGBoost Regressor** at each assignment stage produces superior, more stable predictions compared to individual base models, equal-weight averaging, and conventional fixed stacking with an out-of-fold linear regression meta-learner.

---

## 🌟 Key Research Highlights

1. **Monotonic Error Reduction Across Stages:**
   - RMSE decreases from **$20.78$** at Stage 1 (after Assignment 1) down to **$19.12$** at Stage 5 (full semester), with $R^2$ growing from **$0.08$** to **$0.22$**.
2. **Dynamic Weight Phase Transition (SLSQP Constrained Optimization):**
   - **Early Stages (1 & 2):** **XGBoost dominates** ($75.4\% \to 53.1\%$), supported by KNN ($22.7\% \to 31.9\%$). SVR is suppressed ($1.9\%$). Tree-based splitting handles noisy, zero-inflated early interaction patterns best.
   - **Late Stages (4 & 5):** **SVR takes the lead** ($56.3\% \to 63.4\%$), while KNN shrinks to $2.9\%$. Once cumulative behaviors stabilize across 40–50 problems, kernel methods excel.
3. **Statistical Validity:**
   - 10 repeated experimental runs across random seeds with strict student-level isolation (70% Train, 15% Val, 15% Test).
   - Zero data leakage: scalers, imputers, and meta-models fitted strictly on training data only.

---

## 🚀 Interactive Web UI Dashboard

An interactive **Streamlit** dashboard is included for simulating student performance forecasts in real time:

- **📂 Test with Sample / Uploaded CSV:** Test custom student spreadsheets or select from built-in sample cohorts.
- **✍️ Single Student Predictor:** Input custom programming values and inspect model breakdown charts.
- **👤 Real CodeWorkout Cohort Explorer:** Trace any of the 878 students longitudinally from Assignment 1 to 5.
- **📊 Benchmark & Weight Evolution:** Visualizations of learned weights and metric tables.

To run the web app:
```bash
python -m streamlit run app.py
```

---

## 🛠️ Project Structure

```text
├── config.py                               # Global project configuration
├── requirements.txt                        # Python dependencies
├── main.py                                 # Single-command end-to-end pipeline runner
├── app.py                                  # Interactive Streamlit Web UI
├── .gitignore
│
├── data/
│   ├── raw/                                # Raw unzipped datasets (S19 & F19)
│   ├── processed/                          # 5 cumulative stage parquet files
│   └── splits/                             # 10 student-level split JSON files
│
├── sample_data/                            # Ready-made demo CSVs for testing the UI
│   ├── mixed_cohort_case_study.csv
│   ├── new_students_stage_1.csv
│   ├── new_students_stage_3.csv
│   └── new_students_stage_5.csv
│
├── src/
│   ├── preprocessing.py                    # Data cleaning, deduplication, target unification
│   ├── feature_engineering.py              # Extraction of 7 features across 5 stages
│   ├── splits.py                           # Student-level 70/15/15 stratified partitioning
│   ├── models.py                           # KNN, SVR, and XGBoost training with 5-fold CV
│   ├── weight_optimizer.py                 # Constrained SLSQP optimizer
│   ├── ensemble.py                         # Baselines: Equal-Weight & Fixed Stacking (OOF)
│   ├── evaluation.py                       # Paired t-tests and statistical evaluation
│   └── plots.py                            # Generation of 7 publication-grade figures
│
├── models/saved/                           # Serialized production models & scalers
└── results/
    ├── predictions/                        # 50 JSON prediction records (5 stages x 10 seeds)
    ├── weights/                            # stage_adaptive_weights.csv
    ├── metrics/                            # Consolidated metrics & significance tests
    └── plots/                              # 7 high-resolution 300 DPI publication plots
```

---

## 💻 Quick Start & Reproduction

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full Machine Learning Pipeline (Phases 1-10)
```bash
python main.py
```

### 3. Launch the Interactive Web Dashboard
```bash
python -m streamlit run app.py
```

---

## 📈 The Seven Extracted Programming Pattern Features

1. `TimeSpent`: Total active problem-solving time (10-minute idle capped).
2. `Valid`: Count of executable, syntactically valid submissions.
3. `CorrectSub`: Submissions passing all unit test cases (score = 1.0).
4. `IncorrectSub`: Valid submissions that failed one or more unit tests.
5. `CompileError`: Syntax and compilation errors encountered.
6. `Scores`: Mean highest score attained per problem across cumulative assignments.
7. `EditDistance`: Cumulative normalized Levenshtein distance between consecutive code states.
