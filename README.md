# AtmosSense 🌦️

**AtmosSense** is a full-stack Weather Intelligence web application built with Django and PyTorch. It predicts real-time weather, future temperature trends, and rainfall probability using deep learning.

## Features
- **Real-Time Weather Data**: Integrated with the OpenWeatherMap API to fetch current metrics.
- **Glassmorphism UI**: A beautiful, modern, and highly responsive frontend interface.
- **Dynamic Charts**: Interactive data visualization using Chart.js to map predicted temperature trends.
- **Transfer Learning (Neural Networks)**:
  - We pre-trained a **PyTorch Autoencoder** to learn fundamental weather representations from 5,000 synthetically generated historical weather records.
  - The encoder layers were **frozen**, and new specialized heads were fine-tuned for two tasks:
    - **Rain Classification** (Binary Cross-Entropy)
    - **Temperature Regression** (Mean Squared Error)

## Tech Stack
- **Backend**: Python, Django
- **Frontend**: HTML5, Vanilla CSS, JavaScript, Chart.js
- **Machine Learning**: PyTorch, Scikit-Learn (for data scaling), Pandas, Numpy

## Setup Instructions

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
Visit `http://127.0.0.1:8000/` in your browser!

## Project Structure
- `data/`: Contains the script to generate synthetic weather data.
- `ml_models/`: Contains the PyTorch `.pth` weights, the `scaler.pkl`, and the training scripts.
- `weather_app/`: The main Django application containing views, models, and ML inference logic.
- `atmossense_project/`: Django configuration and routing.
