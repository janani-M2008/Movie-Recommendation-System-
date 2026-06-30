# Movie Recommendation System (Collaborative Filtering)

A recommendation engine built on the MovieLens dataset using both
memory-based (item-item KNN) and model-based (SVD matrix factorization)
collaborative filtering, with a live Streamlit demo.

## Project structure
```
movie-recommender/
├── data/                  # raw + processed data (not committed, see download script)
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
├── src/
│   ├── data_loader.py     # download + load + clean MovieLens data
│   ├── matrix_builder.py  # build user-item sparse matrix
│   ├── cf_models.py       # item-item KNN + SVD models
│   ├── evaluate.py        # RMSE, Precision@K, Recall@K
│   └── recommend.py       # generate top-N recommendations for a user
├── app/
│   └── app.py             # Streamlit demo
├── requirements.txt
└── README.md
```

## Setup
```bash
git clone <your-repo-url>
cd movie-recommender
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/data_loader.py   # downloads MovieLens 100k dataset
```

## Train & evaluate
```bash
python src/cf_models.py
python src/evaluate.py
```

## Run the demo
```bash
streamlit run app/app.py
```

## Approach
- **Data**: MovieLens 100k (100,000 ratings, 943 users, 1,682 movies)
- **Models compared**:
  - Item-item KNN (cosine similarity)
  - SVD matrix factorization (via `surprise`)
  - Popularity baseline
- **Metrics**: RMSE, MAE, Precision@10, Recall@10
- **Cold start**: falls back to popularity-based recommendations for new users

## Results
| Model            | RMSE | Precision@10 |
|------------------|------|--------------|
| Popularity baseline | -    | -            |
| Item-item KNN    | -    | -            |
| SVD              | -    | -            |

*(fill in after running `src/evaluate.py`)*

## Next steps
- Add content-based features (genre, tags) for a hybrid model
- Deploy demo to Streamlit Cloud / Hugging Face Spaces
