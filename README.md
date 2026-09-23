# Hypertension Risk Predictor (Vercel Version)

Modern web app built for **Vercel**:
- Frontend: Next.js 14 + Tailwind CSS
- Backend: Python Serverless Function
- Model: Random Forest trained on NHANES 2017–2018

## Project Structure

```
vercel-hypertension/
├── app/
│   ├── page.tsx                 # Main UI
│   ├── layout.tsx
│   ├── globals.css
│   └── api/predict/route.py     # Python prediction endpoint
├── model/
│   └── hypertension_random_forest.pkl
├── package.json
├── requirements.txt
├── vercel.json
└── README.md
```

## Deploy to Vercel (Step-by-step)

### 1. Push to GitHub
Create a new GitHub repository and upload all files from this folder.

### 2. Import Project on Vercel
1. Go to [https://vercel.com](https://vercel.com) and sign in
2. Click **Add New… → Project**
3. Import your GitHub repository
4. Framework Preset should detect **Next.js**
5. Leave Build settings as default
6. Click **Deploy**

### 3. Important Notes about the Python Function
- Vercel supports Python serverless functions
- The model file (~7.8 MB) + scikit-learn dependencies make the function relatively large
- First request (cold start) may take several seconds
- If you hit size limits, consider:
  - Hosting the model on an external storage and downloading it
  - Converting the model to ONNX for a lighter runtime

### 4. Local Development

```bash
# Install frontend dependencies
npm install

# Run Next.js
npm run dev
```

The Python function will only work fully after deploying to Vercel (or using `vercel dev`).

## Features
- Clean modern UI
- 7 predictors (Age, Sex, BMI, Cholesterol, Diabetes, Smoking, Physical Activity)
- Prediction + confidence score
- Probability breakdown bars
- Strong medical disclaimer

## Medical Disclaimer
This is a **research / educational tool only**.  
It does **not** provide a medical diagnosis and must never replace professional medical advice.
