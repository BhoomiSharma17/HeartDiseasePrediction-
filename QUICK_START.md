# Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python run.py
```

That's it! The app will open in your browser at `http://localhost:8501`

---

## 📋 What Happens When You Run

1. **Checks for trained model**
   - If model exists: Launches app immediately
   - If not: Trains model first (takes 2-3 minutes)

2. **Launches Streamlit app**
   - Opens browser automatically
   - Ready to make predictions

---

## 🎯 Using the Application

### Make a Prediction
1. Navigate to **"🏥 Prediction"** tab
2. Enter patient health metrics
3. Click **"Predict Heart Disease Risk"**
4. View results and confidence score

### Explore Data
1. Go to **"📊 Data Analysis"** tab
2. View dataset statistics
3. Explore interactive visualizations
4. Analyze feature correlations

### View Model Info
1. Check **"ℹ️ Model Info"** tab
2. See model performance metrics
3. Review training details

---

## 🎨 Customize Theme

Toggle between **Dark** and **Light** themes:
- Click theme toggle in sidebar
- Preference is saved automatically

---

## 🧪 Run Tests

```bash
pytest tests/
```

---

## 🐳 Run with Docker

```bash
docker-compose up
```

Access at: `http://localhost:8501`

---

## 📝 Manual Training

If you want to retrain the model:

```bash
python src/model_trainer.py
```

---

## 🆘 Troubleshooting

### Port Already in Use
```bash
streamlit run app/streamlit_app.py --server.port=8502
```

### Model Not Found
```bash
python src/model_trainer.py
```

### Missing Dependencies
```bash
pip install -r requirements.txt --upgrade
```

---

## 📚 Next Steps

- Read [README.md](README.md) for full documentation
- Check [DEPLOYMENT.md](DEPLOYMENT.md) for deployment options
- See [TODO.md](TODO.md) for enhancement ideas
- Review [PORTFOLIO.md](PORTFOLIO.md) for portfolio tips

---

**Need Help?** Open an issue on GitHub or check the documentation!
