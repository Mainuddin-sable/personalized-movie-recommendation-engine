
# Personalized Movie Recommendation Engine

A machine learning-based movie recommendation system that suggests movies to users using Content-Based Filtering, Collaborative Filtering, and Hybrid Recommendation techniques.

The project uses the MovieLens 100K dataset and provides an interactive web application built with Streamlit.


## Features

- Personalized movie recommendations
- Content-Based Filtering
- Collaborative Filtering
- Hybrid Recommendation System
- Popular Movies section
- Movie Search functionality
- Interactive Streamlit dashboard
- Model performance evaluation


## Dataset

This project uses the MovieLens 100K dataset.

- Total ratings: 100,000
- Total users: 943
- Total movies: 1,682

The dataset contains user ratings and movie information used to build and evaluate the recommendation algorithms.



## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- Streamlit
- Plotly
- Joblib


## Recommendation Algorithms

This project implements four recommendation approaches:

1. **Content-Based Filtering:** Recommends movies based on movie genres and similarity.
2. **Collaborative Filtering:** Recommends movies using similarities between users' rating patterns.
3. **Hybrid Recommendation:** Combines Content-Based and Collaborative Filtering.
4. **Popularity-Based Recommendation:** Recommends movies based on their popularity among users.



## Model Evaluation

The recommendation algorithms were evaluated using Precision@10 and Recall@10 on 920 users.

| Algorithm | Precision@10 | Recall@10 |
|---|---:|---:|
| Content-Based | 2.26% | 2.39% |
| Collaborative | 1.00% | 1.81% |
| Hybrid | 4.39% | 5.05% |
| Popularity-Based | 13.58% | 13.72% |

### Evaluation Findings

- Popularity-Based achieved the highest Precision@10 and Recall@10.
- Hybrid performed best among the three personalized approaches.
- Content-Based outperformed Collaborative Filtering in this evaluation.

These results represent the current experimental performance of the system.



## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Mainuddin-sable/personalized-movie-recommendation-engine.git
```

### 2. Navigate to the project folder

```bash
cd Recommendation_Engine
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment (Windows)

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install the required libraries

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
python -m streamlit run app.py
```


## Project Structure

```text
Recommendation_Engine/
│
├── app.py
├── recommender.py
├── styles.css
├── requirements.txt
├── README.md
│
├── models/
│   └── recommendation_engine.pkl

```


## Future Improvements

- Improve recommendation accuracy.
- Experiment with advanced recommendation algorithms.
- Add movie posters and additional movie information.
- Deploy the application online.

## Live Demo

Try the application here:

[Open CineMatch Movie Recommendation Engine](https://personalized-movie-recommendation-engine-2g7t3rrpdhltegu4n2yyj.streamlit.app/)
