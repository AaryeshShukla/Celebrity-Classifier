# Celebrity-Classifier

Machine Learning powered web app that classifies celebrity faces from uploaded images using SVM, wavelet feature extraction, and a custom-built dataset.

## Run locally

```powershell
pip install -r requirements.txt
python server/server.py
```

Open `http://127.0.0.1:5000` in your browser.

## Deploy to Vercel

This repo includes `vercel.json`, `requirements.txt`, and `.vercelignore`.

```powershell
vercel
```

The frontend calls `/classify_image`, so it works on both localhost and the deployed Vercel URL.
