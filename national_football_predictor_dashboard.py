import streamlit as st
import base64
import pandas as pd

    
# ----- Page configs (tab title, favicon) -----
st.set_page_config(
    page_title="National Football Predictor",
    page_icon="⚽️",
)


# ----- Left menu -----
with st.sidebar:
    st.header("National Football Team Prediction Tool")
    st.write("###")
    st.write("**Author:** Christopher Windsor (https://github.com/windsrc)")


# ----- Top title -----
st.write(f"""<div style="text-align: center;"><h1 style="text-align: center;">Predict a match!</h1></div>""", unsafe_allow_html=True)

# ----- Loading the dataset -----

@st.cache_data
def load_data():
    data_path = "data/results.csv"

    df_results = pd.read_csv(data_path) 

    if df_results is not None:
        df_results["date"] = pd.to_datetime(df_results["Date"]).dt.date

    return df_results  # a Pandas DataFrame


df_results = load_data()

unique_team_list = df_results["team"].unique()


# ----- Select the team -----
if unique_team_list is not None:
    # Getting the list of teams to compare from the user
    selected_teams = st.multiselect("Select the teams for the match: ", unique_team_list, default=["Austria", "Germany"], max_selections=2)
else:
    st.subheader("⚠️ Currently unavailable!")

team1 = selected_teams[0]
team2 = selected_teams[1]


# ----- Make calculations -----

# Load the prediction model
with open('model/trained_multi_output_model.pkl', 'rb') as file:
    trained_multi_output_model = pickle.load(file)

# Get the latest team data
def get_latest_team_data(team_name):
        # Combine filtering and getting the last record into one step
        team_data = df_results[(df_results["team"] == team_name) | (df_results["opponent"] == team_name)].iloc[-1]

        if team_data["team"] == team_name:
            rolling_points = team_data["rolling_points"]
            rolling_net_score = team_data["rolling_net_score"]
            elo_rating = team_data["team_elo_rating"]
            
        else:
            rolling_points = team_data["opponent_rolling_points"]
            rolling_net_score = team_data["opponent_rolling_net_score"]
            elo_rating = team_data["opponent_elo_rating"]
        
        return rolling_points, rolling_net_score, elo_rating

# Merge the data
def create_match(team1,team2):
     # Get the latest data for both teams
    team1_rolling_points, team1_rolling_net_score, team1_elo_rating = get_latest_team_data(team1)
    team2_rolling_points, team2_rolling_net_score, team2_elo_rating = get_latest_team_data(team2)

    rolling_points_diff = team1_rolling_points - team2_rolling_points
    rolling_net_score_diff = team1_rolling_net_score - team2_rolling_net_score
    elo_rating_diff = team1_elo_rating - team2_elo_rating

    input_data = pd.DataFrame([[team1_rolling_points, team1_rolling_net_score, team1_elo_rating, 
                                team2_rolling_points, team2_rolling_net_score, team2_elo_rating,rolling_points_diff,rolling_net_score_diff,elo_rating_diff]],
                              columns=input.columns)
    
    return input_data

# Simulate the match
def ml_simulate_match(input_data):
    # Predict the match outcome
    prediction = trained_multi_output_model.predict(input_data)
    
    # Convert predictions to integers
    prediction_round = prediction.round().astype(int)

    return prediction_round


# ----- Select the team -----
match_data = create_match(team1,team2)
result = ml_simulate_match(match_data)

result_text = f"""
    ### {team1} {result[0][0]} :  {team2} {result[0][1]}
    *Form Advantage: {match_data["rolling_points_diff"][0]}*
    *Current Efficience Advantage: {match_data["rolling_score_diff"][0]}*
    *Quality Advantage: {match_data["elo_diff"][0]}*
    """

st.write(result_text)

