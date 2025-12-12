"""
utils.py - Data Handling & Feature Engineering Module

This module contains utility functions for:
- Fetching data from the srating.io API
- Calculating ELO ratings for teams
- Engineering features for machine learning models
"""

import os
import requests
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta


# ============================================================================
# API INTERACTION FUNCTIONS
# ============================================================================

def fetch_data_from_srating(endpoint: str, params: Optional[Dict] = None) -> Dict:
    """
    Safely connect to the srating.io API and fetch data.
    
    Args:
        endpoint: API endpoint path (e.g., 'game/getGames', 'organization/read')
        params: Dictionary of query parameters
        
    Returns:
        Dictionary containing the API response data
        
    Raises:
        ValueError: If SRATING_API_KEY is not set
        requests.HTTPError: If the API request fails
    """
    # Get API key from environment variable
    api_key = os.getenv('SRATING_API_KEY')
    if not api_key:
        raise ValueError(
            "SRATING_API_KEY environment variable is not set. "
            "Please set it before running this application."
        )
    
    # Construct full URL
    base_url = "https://api.srating.io/v1"
    url = f"{base_url}/{endpoint}"
    
    # Set up headers with API key
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    
    try:
        # Make the API request
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
        # Raise an exception for bad status codes
        response.raise_for_status()
        
        # Parse and return JSON response
        data = response.json()
        
        # Convert dict values to list if response is a dict
        if isinstance(data, dict):
            return list(data.values())
        return data
        
    except requests.exceptions.Timeout:
        raise requests.HTTPError("API request timed out after 30 seconds")
    except requests.exceptions.ConnectionError:
        raise requests.HTTPError("Failed to connect to srating.io API")
    except requests.exceptions.HTTPError as e:
        raise requests.HTTPError(f"API request failed with status {response.status_code}: {e}")
    except ValueError as e:
        raise ValueError(f"Failed to parse API response as JSON: {e}")


# ============================================================================
# ELO RATING CALCULATION
# ============================================================================

def calculate_elo(df: pd.DataFrame, K: int = 20, initial_elo: int = 1500) -> pd.DataFrame:
    """
    Calculate running ELO ratings for all teams in the dataset.
    
    The ELO rating system updates team ratings based on game outcomes,
    with larger updates for unexpected results. Formula:
    
    Expected Score: E_A = 1 / (1 + 10^((R_B - R_A)/400))
    Updated Rating: R'_A = R_A + K * (S_A - E_A)
    
    Where:
    - R_A, R_B are current ratings
    - S_A is actual score (1 for win, 0 for loss)
    - K is the update factor (sensitivity to new results)
    
    Args:
        df: DataFrame with columns ['date', 'home_team', 'away_team', 
            'home_score', 'away_score']
        K: ELO K-factor (higher = more responsive to recent games)
        initial_elo: Starting ELO rating for all teams
        
    Returns:
        DataFrame with additional columns ['elo_home', 'elo_away', 
        'elo_diff', 'home_win']
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Ensure date column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date to process games chronologically
    df = df.sort_values('date').reset_index(drop=True)
    
    # Initialize ELO ratings dictionary for all teams
    elo_ratings = {}
    
    # Lists to store ELO ratings before each game
    elo_home_list = []
    elo_away_list = []
    
    for idx, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        
        # Initialize teams with default ELO if first appearance
        if home_team not in elo_ratings:
            elo_ratings[home_team] = initial_elo
        if away_team not in elo_ratings:
            elo_ratings[away_team] = initial_elo
        
        # Get current ELO ratings (before this game)
        elo_home = elo_ratings[home_team]
        elo_away = elo_ratings[away_team]
        
        # Store pre-game ELO ratings
        elo_home_list.append(elo_home)
        elo_away_list.append(elo_away)
        
        # Calculate expected scores
        expected_home = 1 / (1 + 10 ** ((elo_away - elo_home) / 400))
        expected_away = 1 - expected_home
        
        # Determine actual scores (1 for win, 0.5 for tie, 0 for loss)
        home_score = row['home_score']
        away_score = row['away_score']
        
        if home_score > away_score:
            actual_home = 1.0
            actual_away = 0.0
        elif home_score < away_score:
            actual_home = 0.0
            actual_away = 1.0
        else:
            actual_home = 0.5
            actual_away = 0.5
        
        # Update ELO ratings
        elo_ratings[home_team] = elo_home + K * (actual_home - expected_home)
        elo_ratings[away_team] = elo_away + K * (actual_away - expected_away)
    
    # Add ELO columns to dataframe
    df['elo_home'] = elo_home_list
    df['elo_away'] = elo_away_list
    df['elo_diff'] = df['elo_home'] - df['elo_away']
    
    # Add binary target: 1 if home team won, 0 otherwise
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
    
    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def calculate_rolling_stats(df: pd.DataFrame, team_col: str, 
                           stat_cols: List[str], windows: List[int] = [5, 10]) -> pd.DataFrame:
    """
    Calculate rolling average statistics for teams.
    
    Args:
        df: DataFrame sorted by date
        team_col: Column name for team identifier
        stat_cols: List of columns to calculate rolling stats for
        windows: List of window sizes for rolling calculations
        
    Returns:
        DataFrame with additional rolling average columns
    """
    df = df.copy()
    
    for window in windows:
        for stat in stat_cols:
            col_name = f'{team_col}_{stat}_rolling_{window}'
            df[col_name] = df.groupby(team_col)[stat].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
    
    return df


def calculate_rest_days(df: pd.DataFrame, team_col: str, date_col: str = 'date') -> pd.Series:
    """
    Calculate the number of rest days since the team's last game.
    
    Args:
        df: DataFrame with date and team columns
        team_col: Column name for team identifier
        date_col: Column name for date
        
    Returns:
        Series containing rest days for each game
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Calculate days since last game
    df['prev_game_date'] = df.groupby(team_col)[date_col].shift(1)
    df['rest_days'] = (df[date_col] - df['prev_game_date']).dt.days
    
    # Fill first game with median rest days or 7 if not available
    median_rest = df['rest_days'].median() if df['rest_days'].notna().any() else 7
    df['rest_days'] = df['rest_days'].fillna(median_rest)
    
    return df['rest_days']


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master feature engineering function that creates all features for modeling.
    
    This function orchestrates the creation of various features including:
    - ELO ratings and differentials
    - Rolling averages of team performance
    - Rest days between games
    - Home court advantage indicators
    - Additional domain-specific features
    
    Args:
        df: Raw DataFrame with game data
        
    Returns:
        DataFrame with engineered features ready for modeling
    """
    # Ensure we have a copy
    df = df.copy()
    
    # 1. Calculate ELO ratings (must be done first, chronologically)
    df = calculate_elo(df, K=20)
    
    # 2. Calculate point differential
    df['point_diff'] = df['home_score'] - df['away_score']
    
    # 3. Calculate rolling statistics for home teams
    df = calculate_rolling_stats(
        df, 
        team_col='home_team',
        stat_cols=['home_score', 'point_diff'],
        windows=[3, 5, 10]
    )
    
    # 4. Calculate rolling statistics for away teams
    df = calculate_rolling_stats(
        df,
        team_col='away_team', 
        stat_cols=['away_score'],
        windows=[3, 5, 10]
    )
    
    # 5. Calculate rest days
    df['home_rest_days'] = calculate_rest_days(df, 'home_team')
    df['away_rest_days'] = calculate_rest_days(df, 'away_team')
    df['rest_diff'] = df['home_rest_days'] - df['away_rest_days']
    
    # 6. Create additional features
    
    # Home court advantage (binary)
    df['is_home'] = 1  # All games have a home team
    
    # ELO momentum (change in recent games)
    df['elo_home_momentum'] = df.groupby('home_team')['elo_home'].diff(periods=1).fillna(0)
    df['elo_away_momentum'] = df.groupby('away_team')['elo_away'].diff(periods=1).fillna(0)
    
    # Win streak features (simplified)
    df['home_win_streak'] = df.groupby('home_team')['home_win'].transform(
        lambda x: x.rolling(window=5, min_periods=1).sum()
    )
    
    # 7. Handle any remaining NaN values
    df = df.fillna(df.median(numeric_only=True))
    
    return df


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_feature_columns() -> List[str]:
    """
    Return the list of feature column names used for modeling.
    
    Returns:
        List of feature column names
    """
    return [
        'elo_home',
        'elo_away',
        'elo_diff',
        'home_team_home_score_rolling_3',
        'home_team_home_score_rolling_5',
        'home_team_home_score_rolling_10',
        'home_team_point_diff_rolling_3',
        'home_team_point_diff_rolling_5',
        'home_team_point_diff_rolling_10',
        'away_team_away_score_rolling_3',
        'away_team_away_score_rolling_5',
        'away_team_away_score_rolling_10',
        'home_rest_days',
        'away_rest_days',
        'rest_diff',
        'is_home',
        'elo_home_momentum',
        'elo_away_momentum',
        'home_win_streak'
    ]


def prepare_prediction_features(home_team: str, away_team: str, 
                                historical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for a single game prediction.
    
    Args:
        home_team: Name of the home team
        away_team: Name of the away team
        historical_df: DataFrame with historical games and features
        
    Returns:
        Single-row DataFrame with features for prediction
    """
    # Get most recent stats for both teams
    home_recent = historical_df[historical_df['home_team'] == home_team].iloc[-1] if len(historical_df[historical_df['home_team'] == home_team]) > 0 else None
    away_recent = historical_df[historical_df['away_team'] == away_team].iloc[-1] if len(historical_df[historical_df['away_team'] == away_team]) > 0 else None
    
    # Create feature dictionary
    features = {}
    feature_cols = get_feature_columns()
    
    for col in feature_cols:
        if col.startswith('home_') or col.startswith('elo_home'):
            features[col] = home_recent[col] if home_recent is not None else 0
        elif col.startswith('away_') or col.startswith('elo_away'):
            features[col] = away_recent[col] if away_recent is not None else 0
        elif col == 'elo_diff':
            features[col] = features.get('elo_home', 1500) - features.get('elo_away', 1500)
        else:
            features[col] = 0
    
    return pd.DataFrame([features])


if __name__ == "__main__":
    # Test code for development
    print("Utils module loaded successfully")
    print(f"Feature columns: {get_feature_columns()}")
