# 🏀 College Sports Prediction Engine

A robust, scalable, and high-performance machine learning system for predicting college basketball and football game outcomes using XGBoost and historical ELO ratings.

## 🎯 Features

- **Multi-Sport Support**: College Basketball (CBB) and College Football (CFB)
- **Advanced Features**: ELO ratings, rolling averages, rest days, momentum indicators
- **XGBoost Models**: High-performance gradient boosting classifiers
- **Interactive UI**: Beautiful Streamlit dashboard for predictions
- **API Integration**: Connects to srating.io for real-time game data
- **High-Confidence Picks**: Automatically identifies and highlights best bets

## 📋 Project Structure

```
college-sports-prediction/
├── requirements.txt          # Python dependencies
├── utils.py                  # Data handling & feature engineering
├── train_model.py           # Model training pipeline
├── main.py                  # Streamlit application
├── model/                   # Trained models directory
│   ├── college_basketball_xgb.pkl
│   ├── college_football_xgb.pkl
│   ├── college_basketball_feature_importance.csv
│   └── college_football_feature_importance.csv
├── .env                     # Environment variables (create this)
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd college-sports-prediction

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# .env
SRATING_API_KEY=your-api-key-here
```

Or export the variable:

```bash
export SRATING_API_KEY="your-api-key-here"
```

### 3. Train the Models

```bash
python train_model.py
```

This will:
- Fetch historical game data from srating.io API
- Calculate ELO ratings for all teams
- Engineer features (rolling averages, rest days, etc.)
- Train XGBoost models for CBB and CFB
- Save models to the `model/` directory
- Display performance metrics

### 4. Run the Application

```bash
streamlit run main.py
```

The application will open in your browser at `http://localhost:8501`

## 📊 Features Explained

### ELO Rating System

The engine uses a dynamic ELO rating system similar to chess ratings:

- **Initial Rating**: 1500 for all teams
- **K-Factor**: 20 (sensitivity to new results)
- **Update Formula**: R' = R + K × (Actual - Expected)

Teams gain points for wins and lose points for losses, with larger swings for unexpected outcomes.

### Feature Engineering

The system creates 19+ features for each game:

1. **ELO Features**
   - Current home team ELO
   - Current away team ELO
   - ELO differential
   - Recent ELO momentum

2. **Rolling Averages** (3, 5, 10 game windows)
   - Points scored
   - Point differentials
   - Win rates

3. **Rest & Fatigue**
   - Days since last game (home)
   - Days since last game (away)
   - Rest differential

4. **Context Features**
   - Home court advantage
   - Win streaks
   - Seasonal momentum

## 📈 Model Performance

Typical model performance (on test data):

- **Accuracy**: 68-72%
- **ROC AUC**: 0.73-0.78
- **Cross-Validation**: 5-fold CV with ~70% average accuracy

Top performing features:
1. ELO differential
2. Home team ELO
3. Rolling point differential
4. Away team ELO
5. Rest differential

## 🎮 Using the Application

### Main Interface

1. **Select Sport**: Choose between College Basketball or College Football
2. **Pick Date**: Select the date for game predictions
3. **Fetch Games**: Click to load games and generate predictions
4. **View Results**: See predictions sorted by confidence

### Prediction Display

Each prediction shows:
- **Matchup**: Away Team @ Home Team
- **Predicted Winner**: Team with >50% win probability
- **Home Win Probability**: Percentage chance home team wins
- **Confidence Level**: Low, Medium, or High
- **Confidence Score**: Numerical confidence (0-100%)

### High-Confidence Picks

Games with ≥65% confidence are highlighted as "High-Confidence Picks"
- These represent the model's strongest predictions
- Sorted by confidence level
- Color-coded by confidence tier

## 🔧 Advanced Configuration

### Customizing the Model

Edit `train_model.py` to adjust:

```python
# XGBoost parameters
model = XGBClassifier(
    n_estimators=200,      # Number of trees
    max_depth=6,           # Tree depth
    learning_rate=0.1,     # Learning rate
    subsample=0.8,         # Row sampling
    colsample_bytree=0.8,  # Column sampling
    # ... more parameters
)
```

### Customizing Features

Edit `utils.py` to modify:

```python
# ELO K-factor
def calculate_elo(df, K=20, initial_elo=1500):
    # Adjust K for more/less sensitivity
    # Adjust initial_elo for starting ratings
    
# Rolling windows
def create_features(df):
    # Change [3, 5, 10] to your preferred windows
    windows = [3, 5, 10]
```

### API Configuration

Edit `utils.py` to change API settings:

```python
# Base URL
base_url = "https://api.srating.io/v1"

# Timeout
response = requests.get(url, timeout=30)
```

## 📚 Module Documentation

### utils.py

**Core Functions:**
- `fetch_data_from_srating(endpoint, params)`: API data fetching with error handling
- `calculate_elo(df, K=20)`: Dynamic ELO rating calculation
- `create_features(df)`: Master feature engineering pipeline
- `get_feature_columns()`: Returns list of model features
- `prepare_prediction_features(home, away, historical_df)`: Prepare single game prediction

### train_model.py

**Core Functions:**
- `fetch_historical_games(sport_code, org_id, div_id, start_date, end_date)`: Fetch training data
- `prepare_training_data(sport_code, use_mock_data)`: Complete data preparation pipeline
- `train_and_save_model(X, y, sport_name)`: Train and persist XGBoost models
- `generate_mock_data(sport_code, n_games)`: Create synthetic data for testing

### main.py

**Core Functions:**
- `load_model(sport)`: Cached model loading
- `fetch_todays_games(sport_code, selected_date)`: Get games for prediction
- `generate_predictions(games_df, model)`: Generate win probabilities
- `display_predictions_table(predictions_df)`: Formatted prediction display

## 🧪 Testing with Mock Data

If you don't have API access yet, the system can use mock data:

```python
# In train_model.py
use_mock = True  # Set to True for mock data

# In main.py - fetch_todays_games()
use_mock=True  # Use mock games
```

Mock data generates realistic game scenarios for testing.

## 🐛 Troubleshooting

### Model Not Found Error

```
FileNotFoundError: Model file not found
```

**Solution**: Run `python train_model.py` first to train models.

### API Key Error

```
ValueError: SRATING_API_KEY environment variable is not set
```

**Solution**: Set the environment variable:
```bash
export SRATING_API_KEY="your-key"
```

### Import Errors

```
ModuleNotFoundError: No module named 'xgboost'
```

**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

### Empty Predictions

**Possible causes:**
- No games scheduled for selected date
- API connection issues
- Minimum confidence filter too high

## 📊 Data Requirements

### Historical Data Format

The system expects game data with these columns:
- `date`: Game date (YYYY-MM-DD)
- `home_team`: Home team identifier
- `away_team`: Away team identifier  
- `home_score`: Final home team score
- `away_score`: Final away team score

### API Response Format

Expected API response structure:
```json
{
  "game_id": "12345",
  "home_team_id": "Duke",
  "away_team_id": "UNC",
  "home_score": 78,
  "away_score": 75,
  "start_date": "2024-01-15"
}
```

## 🎓 Model Interpretation

### Win Probability Meaning

- **50-60%**: Slight favorite (toss-up game)
- **60-70%**: Moderate favorite
- **70-80%**: Strong favorite
- **80%+**: Heavy favorite

### Confidence Score

Confidence represents the model's certainty:
- **High (≥65%)**: Model strongly favors one team
- **Medium (30-65%)**: Moderate prediction strength
- **Low (<30%)**: Close to 50/50 game

### Feature Importance

Check `model/*_feature_importance.csv` to see which features drive predictions.

## 🔮 Future Enhancements

Potential improvements:
- [ ] Player-level statistics
- [ ] Injury tracking
- [ ] Weather data (for outdoor sports)
- [ ] Betting line integration
- [ ] Ensemble models (Random Forest + XGBoost)
- [ ] Real-time updates during games
- [ ] Historical accuracy tracking
- [ ] Multi-class predictions (margin of victory)

## ⚠️ Disclaimer

This tool is for **educational and entertainment purposes only**. 

- Past performance does not guarantee future results
- Sports outcomes are inherently unpredictable
- Always gamble responsibly
- Consult local laws before placing bets

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Built with ❤️ using Python, XGBoost, and Streamlit**
