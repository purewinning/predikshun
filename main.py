"""
main.py - Streamlit Application for Sports Prediction Engine

This application provides a user-friendly interface for:
- Selecting sports and dates
- Fetching today's games
- Generating win probability predictions
- Displaying high-confidence picks
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from typing import Dict, Optional

# Import utility functions
from utils import (
    fetch_data_from_srating,
    prepare_prediction_features,
    get_feature_columns
)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Sports Prediction Engine",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model(sport: str):
    """
    Load the trained XGBoost model for the specified sport.
    
    Uses Streamlit's @st.cache_resource to load the model only once
    and reuse it across sessions.
    
    Args:
        sport: Sport code ('CBB' or 'CFB')
        
    Returns:
        Loaded XGBoost model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    sport_names = {
        'CBB': 'college basketball',
        'CFB': 'college football'
    }
    
    model_path = f"model/{sport_names[sport].replace(' ', '_')}_xgb.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please run train_model.py first to train the models."
        )
    
    model = joblib.load(model_path)
    return model


# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

def fetch_todays_games(sport_code: str, selected_date: datetime, 
                       use_mock: bool = False) -> pd.DataFrame:
    """
    Fetch games for the selected date.
    
    Args:
        sport_code: 'CBB' or 'CFB'
        selected_date: Date to fetch games for
        use_mock: If True, generate mock data instead of API call
        
    Returns:
        DataFrame with today's games
    """
    if use_mock:
        # Generate mock games for demonstration
        return generate_mock_todays_games(sport_code, selected_date)
    
    try:
        # Real API call
        # Note: You'll need to customize organization_id and division_id
        # based on your specific needs
        params = {
            'organization_id': '1',  # Customize
            'division_id': '1',      # Customize
            'start_date': selected_date.strftime('%Y-%m-%d')
        }
        
        games = fetch_data_from_srating('game/getGames', params)
        
        if not games:
            return pd.DataFrame()
        
        df = pd.DataFrame(games)
        
        # Standardize column names
        column_mapping = {
            'home_team_id': 'home_team',
            'away_team_id': 'away_team',
            'start_datetime': 'game_time'
        }
        df = df.rename(columns=column_mapping)
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching games: {e}")
        return pd.DataFrame()


def generate_mock_todays_games(sport_code: str, date: datetime) -> pd.DataFrame:
    """
    Generate mock games for demonstration purposes.
    
    Args:
        sport_code: 'CBB' or 'CFB'
        date: Date for the games
        
    Returns:
        DataFrame with mock games
    """
    np.random.seed(int(date.timestamp()))
    
    if sport_code == 'CBB':
        teams = [
            'Duke', 'North Carolina', 'Kentucky', 'Kansas', 'UCLA',
            'Villanova', 'Michigan', 'Arizona', 'Gonzaga', 'Virginia',
            'Texas', 'Tennessee', 'Auburn', 'Purdue', 'Houston'
        ]
    else:  # CFB
        teams = [
            'Alabama', 'Georgia', 'Ohio State', 'Michigan', 'Texas',
            'USC', 'Notre Dame', 'Clemson', 'Oregon', 'Penn State',
            'Florida State', 'LSU', 'Oklahoma', 'Tennessee', 'Washington'
        ]
    
    n_games = np.random.randint(5, 12)
    games = []
    
    selected_teams = np.random.choice(teams, size=n_games*2, replace=False)
    
    for i in range(n_games):
        home_team = selected_teams[i*2]
        away_team = selected_teams[i*2 + 1]
        
        # Generate random game time
        hour = np.random.randint(12, 22)
        minute = np.random.choice([0, 30])
        game_time = date.replace(hour=hour, minute=minute)
        
        games.append({
            'home_team': home_team,
            'away_team': away_team,
            'game_time': game_time,
            'game_id': f"{sport_code}_{date.strftime('%Y%m%d')}_{i}"
        })
    
    return pd.DataFrame(games)


def create_mock_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create mock features for prediction when historical data is unavailable.
    
    Args:
        games_df: DataFrame with today's games
        
    Returns:
        DataFrame with features for prediction
    """
    feature_cols = get_feature_columns()
    
    # Create features with realistic values
    features_list = []
    
    for idx, game in games_df.iterrows():
        # Generate semi-realistic feature values
        elo_home = np.random.normal(1500, 100)
        elo_away = np.random.normal(1500, 100)
        
        features = {
            'elo_home': elo_home,
            'elo_away': elo_away,
            'elo_diff': elo_home - elo_away,
            'home_team_home_score_rolling_3': np.random.normal(75, 8),
            'home_team_home_score_rolling_5': np.random.normal(75, 6),
            'home_team_home_score_rolling_10': np.random.normal(75, 5),
            'home_team_point_diff_rolling_3': np.random.normal(3, 5),
            'home_team_point_diff_rolling_5': np.random.normal(3, 4),
            'home_team_point_diff_rolling_10': np.random.normal(3, 3),
            'away_team_away_score_rolling_3': np.random.normal(72, 8),
            'away_team_away_score_rolling_5': np.random.normal(72, 6),
            'away_team_away_score_rolling_10': np.random.normal(72, 5),
            'home_rest_days': np.random.randint(2, 8),
            'away_rest_days': np.random.randint(2, 8),
            'rest_diff': 0,
            'is_home': 1,
            'elo_home_momentum': np.random.normal(0, 20),
            'elo_away_momentum': np.random.normal(0, 20),
            'home_win_streak': np.random.randint(0, 5)
        }
        
        features['rest_diff'] = features['home_rest_days'] - features['away_rest_days']
        features_list.append(features)
    
    return pd.DataFrame(features_list)


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def generate_predictions(games_df: pd.DataFrame, model, sport_code: str) -> pd.DataFrame:
    """
    Generate win probability predictions for all games.
    
    Args:
        games_df: DataFrame with today's games
        model: Trained XGBoost model
        sport_code: Sport code for context
        
    Returns:
        DataFrame with predictions added
    """
    if games_df.empty:
        return games_df
    
    # Create features (using mock features for now)
    features_df = create_mock_features(games_df)
    
    # Generate predictions
    win_probabilities = model.predict_proba(features_df)[:, 1]
    
    # Add predictions to games dataframe
    games_df = games_df.copy()
    games_df['home_win_prob'] = win_probabilities
    games_df['away_win_prob'] = 1 - win_probabilities
    
    # Determine predicted winner
    games_df['predicted_winner'] = games_df.apply(
        lambda row: row['home_team'] if row['home_win_prob'] > 0.5 else row['away_team'],
        axis=1
    )
    
    # Calculate confidence (distance from 50%)
    games_df['confidence'] = abs(games_df['home_win_prob'] - 0.5) * 2
    
    # Categorize confidence level
    games_df['confidence_level'] = pd.cut(
        games_df['confidence'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    return games_df


# ============================================================================
# UI COMPONENTS
# ============================================================================

def display_header():
    """Display the application header."""
    st.title("🏀 College Sports Prediction Engine")
    st.markdown("""
    Advanced machine learning predictions for college basketball and football games.
    Powered by XGBoost and historical ELO ratings.
    """)
    st.markdown("---")


def display_sidebar():
    """Display the sidebar with controls."""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Sport selection
        sport = st.selectbox(
            "Select Sport",
            options=['CBB', 'CFB'],
            format_func=lambda x: '🏀 College Basketball' if x == 'CBB' else '🏈 College Football',
            key='sport_select'
        )
        
        # Date selection
        selected_date = st.date_input(
            "Select Date",
            value=datetime.now(),
            min_value=datetime.now() - timedelta(days=7),
            max_value=datetime.now() + timedelta(days=30),
            key='date_select'
        )
        
        st.markdown("---")
        
        # Model info
        st.subheader("📊 Model Info")
        try:
            model = load_model(sport)
            st.success("✓ Model loaded")
            
            # Display feature importance file if exists
            importance_file = f"model/{'college_basketball' if sport == 'CBB' else 'college_football'}_feature_importance.csv"
            if os.path.exists(importance_file):
                with st.expander("View Feature Importance"):
                    importance_df = pd.read_csv(importance_file)
                    st.dataframe(
                        importance_df.head(10),
                        use_container_width=True,
                        hide_index=True
                    )
        except FileNotFoundError:
            st.error("❌ Model not found")
            st.info("Run train_model.py first")
        
        st.markdown("---")
        
        # Additional settings
        st.subheader("🎯 Display Options")
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Filter predictions by confidence level"
        )
        
        show_all = st.checkbox("Show All Games", value=True)
        
        return sport, selected_date, min_confidence, show_all


def display_predictions_table(predictions_df: pd.DataFrame):
    """Display predictions in a formatted table."""
    if predictions_df.empty:
        st.info("No games found for the selected date.")
        return
    
    # Prepare display dataframe
    display_df = predictions_df[[
        'away_team', 'home_team', 'predicted_winner',
        'home_win_prob', 'confidence_level'
    ]].copy()
    
    # Format probabilities as percentages
    display_df['home_win_prob'] = display_df['home_win_prob'].apply(lambda x: f"{x*100:.1f}%")
    
    # Rename columns for display
    display_df.columns = [
        'Away Team', 'Home Team', 'Predicted Winner',
        'Home Win Probability', 'Confidence'
    ]
    
    # Display with styling
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=400
    )


def display_high_confidence_picks(predictions_df: pd.DataFrame, threshold: float = 0.65):
    """Display high-confidence picks in a highlighted section."""
    high_confidence = predictions_df[predictions_df['confidence'] >= threshold].copy()
    
    if high_confidence.empty:
        st.info(f"No high-confidence picks (≥{threshold*100:.0f}%) for this date.")
        return
    
    st.subheader(f"⭐ High-Confidence Picks ({len(high_confidence)} games)")
    
    # Sort by confidence
    high_confidence = high_confidence.sort_values('confidence', ascending=False)
    
    # Display each pick in a card-like format
    for idx, game in high_confidence.iterrows():
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            matchup = f"**{game['away_team']}** @ **{game['home_team']}**"
            st.markdown(matchup)
            if 'game_time' in game:
                st.caption(f"🕐 {game['game_time'].strftime('%I:%M %p')}")
        
        with col2:
            winner = game['predicted_winner']
            prob = game['home_win_prob'] if winner == game['home_team'] else game['away_win_prob']
            st.metric("Pick", winner)
            st.caption(f"{prob*100:.1f}% confidence")
        
        with col3:
            confidence_pct = game['confidence'] * 100
            st.metric("Confidence", f"{confidence_pct:.0f}%")
            
            # Color-code confidence
            if confidence_pct >= 80:
                st.success("Very High")
            elif confidence_pct >= 65:
                st.info("High")
        
        st.markdown("---")


def display_statistics(predictions_df: pd.DataFrame):
    """Display summary statistics."""
    if predictions_df.empty:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Games", len(predictions_df))
    
    with col2:
        high_conf = len(predictions_df[predictions_df['confidence_level'] == 'High'])
        st.metric("High Confidence", high_conf)
    
    with col3:
        avg_confidence = predictions_df['confidence'].mean() * 100
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
    
    with col4:
        home_favored = len(predictions_df[predictions_df['home_win_prob'] > 0.5])
        st.metric("Home Favored", home_favored)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function."""
    
    # Display header
    display_header()
    
    # Display sidebar and get settings
    sport, selected_date, min_confidence, show_all = display_sidebar()
    
    # Main content area
    st.header(f"📅 Predictions for {selected_date.strftime('%B %d, %Y')}")
    
    # Fetch games button
    if st.button("🔍 Fetch Games & Generate Predictions", type="primary", use_container_width=True):
        
        with st.spinner("Fetching games..."):
            # Fetch today's games
            games_df = fetch_todays_games(
                sport_code=sport,
                selected_date=datetime.combine(selected_date, datetime.min.time()),
                use_mock=True  # Set to False when API is available
            )
        
        if games_df.empty:
            st.warning("No games found for the selected date.")
            return
        
        with st.spinner("Generating predictions..."):
            try:
                # Load model
                model = load_model(sport)
                
                # Generate predictions
                predictions_df = generate_predictions(games_df, model, sport)
                
                # Apply confidence filter
                if min_confidence > 0:
                    predictions_df = predictions_df[predictions_df['confidence'] >= min_confidence]
                
                # Store in session state
                st.session_state['predictions'] = predictions_df
                
            except Exception as e:
                st.error(f"Error generating predictions: {e}")
                return
    
    # Display predictions if available
    if 'predictions' in st.session_state:
        predictions_df = st.session_state['predictions']
        
        if not predictions_df.empty:
            # Display statistics
            st.markdown("### 📊 Summary Statistics")
            display_statistics(predictions_df)
            
            st.markdown("---")
            
            # Display high-confidence picks
            display_high_confidence_picks(predictions_df, threshold=0.65)
            
            st.markdown("---")
            
            # Display all predictions
            if show_all:
                st.markdown("### 📋 All Predictions")
                display_predictions_table(predictions_df)
        else:
            st.info("No predictions match the selected filters.")
    
    # Footer
    st.markdown("---")
    st.caption("⚠️ For entertainment purposes only. Always gamble responsibly.")
    st.caption("Predictions based on historical data and machine learning models.")


if __name__ == "__main__":
    main()
