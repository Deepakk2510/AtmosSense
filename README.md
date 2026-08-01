# AtmosSense 🌦️

**AtmosSense** is a full-stack Weather Intelligence web application built with Django, PyTorch, and Scikit-Learn. It predicts real-time weather, future temperature trends, and rainfall probability using advanced machine learning architectures.

This project was developed iteratively in two phases to demonstrate the evolution from traditional ensemble methods to deep learning-based Transfer Learning on tabular data.

---

## 🚀 Project Methodology & Pipeline

### 1. Data Generation & Preprocessing
To train our models without relying heavily on rate-limited historical APIs, we generated a highly realistic synthetic dataset (`historical_weather.csv`) comprising 5,000 records. 
- **Features Used:** Temperature, Humidity, Wind Speed, Pressure.
- **Target Variables:** `will_rain` (Boolean), `next_day_temp` (Float).
- **Preprocessing:** We utilized `StandardScaler` from Scikit-Learn to normalize the feature space. This is highly critical for Neural Networks to converge efficiently and for tree-based models to treat features uniformly.

### 2. Phase 1: The Baseline (Random Forest)
Our initial approach relied on traditional, robust ensemble learning architectures via `scikit-learn`. We trained two distinct models:
- **Rain Prediction (Classification)**: We utilized a `RandomForestClassifier` with 100 estimators. Tree-based algorithms naturally handle non-linear tabular patterns well.
- **Temperature Prediction (Regression)**: We utilized a `RandomForestRegressor`.

**Baseline Performance:**
- **Rain Classifier Accuracy**: `78.00%`
- **Temperature Regressor MAE**: `2.40` degrees (Mean Absolute Error)

*Conclusion from Phase 1: Random Forests provided a strong, stable baseline with excellent regression accuracy.*

### 3. Phase 2: The Upgrade (Transfer Learning with PyTorch)
To introduce advanced deep learning techniques, we upgraded the backend predictive engine from Scikit-Learn to a **PyTorch Neural Network**. Because Transfer Learning is typically associated with Computer Vision or NLP, we implemented a highly creative **Tabular Transfer Learning** pipeline:

**Step A: Pre-training the Base Autoencoder**
- We designed a PyTorch Neural Network Autoencoder (`4 -> 16 -> 8 -> 16 -> 4`).
- We trained this network to compress the 4 weather features into an 8-dimensional latent space and reconstruct them, forcing the network to learn the "fundamental laws" of our weather data without any labels.

**Step B: Feature Extraction (Freezing)**
- We extracted the Encoder portion (`4 -> 16 -> 8`) and **froze all of its weights** (`requires_grad = False`). This ensures the learned representations are preserved.

**Step C: Fine-Tuning the Heads**
- **Rain Classifier Head**: We attached a new Dense network with a Sigmoid output to the frozen encoder. We trained *only* this head using Binary Cross-Entropy loss.
- **Temperature Regressor Head**: We attached a separate Dense network with a Linear output to the frozen encoder, training it using Mean Squared Error.

**Transfer Learning Performance:**
- **Transfer Classifier Accuracy**: `79.60%` *(+1.6% Improvement over Random Forest)*
- **Transfer Regressor MAE**: `2.46` degrees *(Comparable to Random Forest)*

*Conclusion from Phase 2: The Neural Network successfully outperformed the Random Forest on classification and remained highly competitive on regression, proving that the frozen Autoencoder successfully learned transferable weather representations.*

---

## 💻 Tech Stack & Architecture
- **Backend Framework**: Python, Django (REST API & Routing)
- **Frontend Design**: HTML5, Vanilla CSS (Glassmorphism aesthetics), JavaScript
- **Data Visualization**: Chart.js (Dynamic temperature forecasting curves)
- **Machine Learning Engine**: PyTorch (Neural Networks), Scikit-Learn (Preprocessing & Baselines), Pandas, Numpy

---

## 🛠️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/YourUsername/AtmosSense.git
cd AtmosSense
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Variables
Create a `.env` file in the root directory and add your OpenWeatherMap API key:
```env
OPENWEATHERMAP_API_KEY=your_api_key_here
```

### 5. Run the Server
```bash
python manage.py runserver
```
Visit `http://127.0.0.1:8000/` in your browser to experience AtmosSense!
