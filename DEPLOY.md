# Streamlit Deployment

Use these files in your GitHub repository:

- `app.py`
- `requirements.txt`
- `sentiment.csv`
- `.gitignore`

Do not upload:

- `venv/`
- `trades.csv`
- `trades.zip`
- `trades (2).zip`

`trades.csv` is intentionally ignored because it is large. The app can load it
in one of three ways:

- Upload it in the Streamlit sidebar after the app starts.
- Keep `trades.csv` locally when running on your own computer.
- Host it somewhere else and add its direct CSV link as a Streamlit secret named
  `TRADES_CSV_URL`.

## Streamlit Community Cloud

1. Push this folder to GitHub.
2. Go to Streamlit Community Cloud.
3. Choose the GitHub repository.
4. Set the main file path to `app.py`.
5. Deploy.
6. Open the app and upload `trades.csv` in the sidebar.

If you prefer automatic loading, host the CSV somewhere that provides a direct
download link, then add this in Streamlit Cloud under app settings > Secrets:

```toml
TRADES_CSV_URL = "https://example.com/trades.csv"
```

## Local Run

If Python is installed locally, run:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

If `python`, `pip`, or `streamlit` are not recognized, reinstall Python and select the option that adds Python to PATH.
