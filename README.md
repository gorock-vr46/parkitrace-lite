# ParkiTrace Lite – runs fully in the browser (free Vercel hosting, zero server RAM)

The AI model runs on the visitor's device (ONNX Runtime Web), so Vercel only serves static files.

## Step 1 – Train and export the light model (on your own computer, once)
    python -m venv venv
    venv\Scripts\activate                (Linux/Mac: source venv/bin/activate)
    pip install -r requirements-train.txt
    python make_demo_data.py             (ONLY for a test; use your real dataset instead)
    python train_light.py --data dataset --epochs 15
Real dataset layout: dataset/healthy/*.png and dataset/parkinson/*.png
Result: model/parkitrace.onnx (about 17 MB) and model/meta.json (accuracy, precision, recall, F1, confusion matrix).

## Step 2 – Test locally
    python -m http.server 8000
Open http://localhost:8000 (do not double-click index.html; the model must be loaded over http).

## Step 3 – Put it on GitHub
Upload every file in this folder INCLUDING the model folder with parkitrace.onnx and meta.json.

## Step 4 – Deploy on Vercel (free)
1. vercel.com -> sign in with GitHub -> Add New -> Project -> pick the repository.
2. Framework Preset: Other. Leave build settings empty. Click Deploy.
3. Open the https link. Phone camera works on https.
