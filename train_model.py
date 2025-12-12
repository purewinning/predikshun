"""
train_model.py - Model Training Pipeline

This module handles:
- Fetching historical data from the srating.io API
- Preparing training data with feature engineering
- Training XGBoost models for game prediction
- Model evaluation and persistence
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib

# Import our utility functions
from utils import (
    fetch_data_from_srating,
    calculate_elo,
    create_features,
    get_feature_columns
)


# ============================================================================
# DATA PREPARATION FUNCTIONS
# ============================================================================

def fetch_historical_games(sport_code: str, organization_id: str, 
                          division_id: str, start_date: str, 
                          end_date: str) -> pd.DataFrame:
    """
    Fetch historical game data from srating.io API for a date range.
    
    Args:
        sport_code: Sport code (e.g., 'CBB', 'CFB')
        organization_id: Organization ID from the API
        division_id: Division ID from the API
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        DataFrame containing historical games
    """
    all_games = []
    
    # Convert dates to datetime objects
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    print(f"Fetching historical data for {sport_code} from {start_date} to {end_date}...")
    
    # Fetch data in chunks (e.g., weekly) to avoid overwhelming the API
    current_date = start
    while current_date <= end:
        try:
            # Fetch games for this date
            params = {
                'organization_id': organization_id,
                'division_id': division_id,
                'start_date': current_date.strftime('%Y-%m-%d')
            }
            
            games = fetch_data_from_srating('game/getGames', params)
            
            if games:
                all_games.extend(games)
                print(f"  Fetched {len(games)} games for {current_date.strftime('%Y-%m-%d')}")
            
        except Exception as e:
            print(f"  Warning: Failed to fetch data for {current_date.strftime('%Y-%m-%d')}: {e}")
        
        # Move to next date
        current_date += timedelta(days=1)
    
    print(f"Total games fetched: {len(all_games)}")
    
    # Convert to DataFrame
    if not all_games:
        raise ValueError(f"No historical data found for {sport_code}")
    
    df = pd.DataFrame(all_games)
    return df


def prepare_training_data(sport_code: str, use_mock_data: bool = False) -> tuple:
    """
    Prepare training data by fetching historical games and engineering features.
    
    This function:
    1. Fetches historical data from the API (or uses mock data)
    2. Calculates ELO ratings
    3. Engineers features
    4. Prepares feature matrix (X) and target vector (y)
    
    Args:
        sport_code: 'CBB' for College Basketball or 'CFB' for College Football
        use_mock_data: If True, generates mock data instead of API calls
        
    Returns:
        Tuple of (X, y, feature_names, full_df) where:
        - X: Feature matrix (numpy array)
        - y: Target vector (numpy array)
        - feature_names: List of feature column names
        - full_df: Full DataFrame with all features (for later reference)
    """
    
    if use_mock_data:
        print(f"\n{'='*60}")
        print(f"Generating mock data for {sport_code}...")
        print(f"{'='*60}\n")
        df = generate_mock_data(sport_code, n_games=1000)
    else:
        # Real API implementation
        print(f"\n{'='*60}")
        print(f"Fetching real data for {sport_code} from srating.io API...")
        print(f"{'='*60}\n")
        
        # Fetch organizations and divisions (you'll need to customize these IDs)
        # For production, you would fetch these dynamically
        if sport_code == 'CBB':
            # College Basketball parameters (customize based on API)
            org_id = "1"  # Replace with actual organization ID
            div_id = "1"  # Replace with actual division ID
        elif sport_code == 'CFB':
            # College Football parameters (customize based on API)
            org_id = "2"  # Replace with actual organization ID
            div_id = "1"  # Replace with actual division ID
        else:
            raise ValueError(f"Unknown sport code: {sport_code}")
        
        # Fetch last 2 seasons of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # ~2 years
        
        df = fetch_historical_games(
            sport_code=sport_code,
            organization_id=org_id,
            division_id=div_id,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
    
    # Clean and standardize the data
    df = clean_game_data(df)
    
    # Engineer features
    print("\nEngineering features...")
    df = create_features(df)
    
    # Get feature columns
    feature_cols = get_feature_columns()
    
    # Prepare X and y
    X = df[feature_cols].values
    y = df['home_win'].values
    
    print(f"\nTraining data prepared:")
    print(f"  Total games: {len(df)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Home win rate: {y.mean():.3f}")
    
    return X, y, feature_cols, df


def clean_game_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize game data from the API.
    
    Args:
        df: Raw DataFrame from API
        
    Returns:
        Cleaned DataFrame with standardized columns
    """
    # Standardize column names (adjust based on actual API response)
    column_mapping = {
        'home_team_id': 'home_team',
        'away_team_id': 'away_team',
        'home_team_score': 'home_score',
        'away_team_score': 'away_score',
        'start_date': 'date',
        'start_datetime': 'date'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Ensure required columns exist
    required_cols = ['date', 'home_team', 'away_team', 'home_score', 'away_score']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Remove incomplete games (games without scores)
    df = df.dropna(subset=['home_score', 'away_score'])
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    return df


def generate_mock_data(sport_code: str, n_games: int = 1000) -> pd.DataFrame:
    """
    Generate realistic mock data for testing when API is unavailable.
    
    Args:
        sport_code: 'CBB' or 'CFB'
        n_games: Number of games to generate
        
    Returns:
        DataFrame with mock game data
    """
    np.random.seed(42)
    
    # Generate team names
    if sport_code == 'CBB':
        teams = [f"CBB_Team_{i}" for i in range(1, 51)]  # 50 teams
        avg_score = 75
        score_std = 12
    else:  # CFB
        teams = [f"CFB_Team_{i}" for i in range(1, 51)]  # 50 teams
        avg_score = 28
        score_std = 10
    
    # Generate dates over 2 seasons
    start_date = datetime(2022, 11, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_games)]
    
    # Generate games
    games = []
    for date in dates:
        # Randomly select home and away teams
        home_team, away_team = np.random.choice(teams, size=2, replace=False)
        
        # Generate scores with some correlation (better teams score more)
        home_base = np.random.normal(avg_score, score_std)
        away_base = np.random.normal(avg_score, score_std)
        
        # Add home court advantage
        home_advantage = np.random.normal(3, 2)
        
        home_score = max(0, int(home_base + home_advantage))
        away_score = max(0, int(away_base))
        
        games.append({
            'date': date,
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score
        })
    
    return pd.DataFrame(games)


# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def train_and_save_model(X: np.ndarray, y: np.ndarray, sport_name: str, 
                        feature_names: list) -> dict:
    """
    Train an XGBoost classifier and save it to disk.
    
    Args:
        X: Feature matrix
        y: Target vector
        sport_name: Name of the sport (for file naming)
        feature_names: List of feature names
        
    Returns:
        Dictionary containing model metrics
    """
    print(f"\n{'='*60}")
    print(f"Training XGBoost model for {sport_name}")
    print(f"{'='*60}\n")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} games")
    print(f"Test set: {len(X_test)} games\n")
    
    # Initialize XGBoost classifier with optimized parameters
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Train the model
    print("Training model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n{'='*60}")
    print("Model Performance")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Away Win', 'Home Win']))
    
    # Cross-validation scores
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance
    print("\nTop 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:<40} {row['importance']:.4f}")
    
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Save the model
    model_filename = f"model/{sport_name.lower()}_xgb.pkl"
    joblib.dump(model, model_filename)
    print(f"\n✓ Model saved to: {model_filename}")
    
    # Save feature importance
    importance_filename = f"model/{sport_name.lower()}_feature_importance.csv"
    feature_importance.to_csv(importance_filename, index=False)
    print(f"✓ Feature importance saved to: {importance_filename}")
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'cv_scores': cv_scores,
        'feature_importance': feature_importance
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to train models for both CBB and CFB.
    """
    print("\n" + "="*60)
    print("SPORTS PREDICTION ENGINE - MODEL TRAINING")
    print("="*60)
    
    # Check if API key is set (if not using mock data)
    use_mock = True  # Set to False when you have API access
    
    if not use_mock and not os.getenv('SRATING_API_KEY'):
        print("\n⚠️  WARNING: SRATING_API_KEY not set!")
        print("Set it using: export SRATING_API_KEY='your-api-key'")
        print("Using mock data instead...\n")
        use_mock = True
    
    # Define sports to train
    sports = [
        {'code': 'CBB', 'name': 'College Basketball'},
        {'code': 'CFB', 'name': 'College Football'}
    ]
    
    results = {}
    
    # Train model for each sport
    for sport in sports:
        try:
            # Prepare training data
            X, y, feature_names, df = prepare_training_data(
                sport_code=sport['code'],
                use_mock_data=use_mock
            )
            
            # Train and save model
            metrics = train_and_save_model(
                X=X,
                y=y,
                sport_name=sport['name'],
                feature_names=feature_names
            )
            
            results[sport['code']] = {
                'success': True,
                'metrics': metrics
            }
            
        except Exception as e:
            print(f"\n❌ Error training {sport['name']} model: {e}")
            results[sport['code']] = {
                'success': False,
                'error': str(e)
            }
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    for sport_code, result in results.items():
        print(f"\n{sport_code}:")
        if result['success']:
            metrics = result['metrics']
            print(f"  ✓ Success")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  ROC AUC: {metrics['auc']:.4f}")
            print(f"  CV Score: {metrics['cv_scores'].mean():.4f}")
        else:
            print(f"  ❌ Failed: {result['error']}")
    
    print("\n" + "="*60)
    print("Training complete! Models saved in ./model/ directory")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
