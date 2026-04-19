# Streamlit Deployment Guide

## 1. Keep These Files In The GitHub Repo

- `app.py`
- `nids_project.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `models/nids_model.pkl`
- `models/nids_preprocessor.pkl`
- `models/nids_metrics.json`
- `data/KDDTrain_.txt` and `data/KDDTest_.txt` if you want online retraining

Do not upload `venv/`, `__pycache__/`, or `data/nids_logs.db`.

## 2. Test Locally First

```powershell
cd F:\nids_complete_project\nids_project
streamlit run app.py
```

Open `http://localhost:8501` and confirm the classifier works.

## 3. Push To GitHub

Create a new GitHub repository, then upload or push the project files.

If using Git commands:

```powershell
git init
git add .
git commit -m "Deploy NIDS Streamlit app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

## 4. Deploy On Streamlit Community Cloud

1. Go to `https://share.streamlit.io`
2. Sign in with GitHub
3. Click `New app`
4. Select your repository
5. Select branch `main`
6. Set main file path to `app.py`
7. Open `Advanced settings`
8. Set Python version to `3.11`
9. Click `Deploy`

Important: Streamlit Community Cloud does not reliably use `runtime.txt` for Python version selection. Set Python `3.11` from Advanced settings or from the deployed app's settings page.

## 5. Database Note

The deployed app uses SQLite by default. On Streamlit Community Cloud, SQLite files are temporary, so saved logs may reset when the app restarts.

For permanent cloud storage, connect the app to an external database such as MySQL, PostgreSQL, or Supabase.
