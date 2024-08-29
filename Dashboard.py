import streamlit as st
import pandas as pd
import pickle
import numpy as np

    
# ----- Page configs (tab title, favicon) -----
st.set_page_config(
    page_title="National Football Predictor",
    page_icon="⚽️",
)


# ----- Left menu -----
with st.sidebar:
    st.image("nationalfootballpredictor_header.jpg", width=300)
    st.header("National Football Team Prediction Tool")
    st.write("###")
    st.write("Interactive project for the EURO 2024 that uses a machine learning model to predict football match outcomes. Users can select two teams to compare, and the app calculates and displays the likelihood of a win, tie, or loss based on historical performance data. The tool generates fair betting odds and visualizes predictions, providing an engaging way to explore match forecasts and betting insights.")
    st.write("**Author:** [Christopher Windsor](https://github.com/windsrc)")


# ----- Top title -----
st.write(f"""<div style="text-align: center;"><h1 style="text-align: center;">Predict a match!</h1></div>""", unsafe_allow_html=True)


# ----- Loading the dataset -----
@st.cache_data
def load_data():
    data_path = "data/modified_results.csv"

    df_results = pd.read_csv(data_path) 

    return df_results  # a Pandas DataFrame

df_results = load_data()

# Displaying the dataset in a expandable table
#with st.expander("Check the complete dataset:"):
#    st.dataframe(df_results)

unique_team_list = df_results["team"].unique()
selected_teams = None

# ----- Select the team -----
if unique_team_list is not None:
    # Getting the list of teams to compare from the user
    selected_teams = st.multiselect("Select the teams for the match: ", unique_team_list, default=["Austria", "Germany"], max_selections=2)
else:
    st.subheader("⚠️ Currently unavailable!")

if len(selected_teams) < 2:
    st.write("Select 2 teams!")
else:
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
                                    columns=['rolling_points', 'rolling_net_score', 'team_elo_rating', 'opponent_rolling_points', 'opponent_rolling_net_score',
                                    'opponent_elo_rating', 'rolling_points_diff', 'rolling_score_diff', 'elo_diff'], dtype='object')
        
        return input_data

    # Simulate the match
    def ml_simulate_match(input_data):
        # Predict the match outcome
        prediction = trained_multi_output_model.predict(input_data)
        
        # Convert predictions to integers
        prediction_round = prediction.round().astype(int)

        return prediction_round
    
    # Define the sigmoid function
    def sigmoid(x, L ,x0, k, b):
        y = L / (1 + np.exp(-k*(x-x0))) + b
        return y

    # Simulate the probablities regarding the distribution of the score differences
    def get_points_probabilities(new_data):
        win_predictions_points_diff = sigmoid(new_data, 1.0219911767395384, 3.0710125946286957, 0.17133835987111373, -0.012618327547552427)
        loss_predictions_points_diff = sigmoid(new_data, -1.0173076171750144, -4.1949160596867605, 0.1724123095353443, 1.0073651064741669)
        tie_predictions_points_diff = 1-(win_predictions_points_diff+loss_predictions_points_diff)
        return [win_predictions_points_diff.item(),tie_predictions_points_diff.item(),loss_predictions_points_diff.item()]
    
    def get_score_probabilities(new_data):
        win_predictions_score_diff = sigmoid(new_data, 0.9292197251654829, 5.497483580698807, 0.09485697620314182, 0.025633267508348358)
        loss_predictions_score_diff = sigmoid(new_data, -0.8833313927905777, -6.37494123582646, 0.10220264014360983, 0.908659500577311)
        tie_predictions_score_diff = 1-(win_predictions_score_diff+loss_predictions_score_diff)
        return [win_predictions_score_diff.item(),tie_predictions_score_diff.item(),loss_predictions_score_diff.item()]
    
    def get_elo_probabilities(new_data):
        win_predictions_elo_diff = sigmoid(new_data, 0.963690164672331, 47.89567169440476, 0.005767877565037683, -0.021828821104679156)
        loss_predictions_elo_diff = sigmoid(new_data, -0.9412743294711561, -89.91127923351979, 0.0059683018353539515, 0.9341988086163746)
        tie_predictions_elo_diff = 1-(win_predictions_elo_diff+loss_predictions_elo_diff)
        return [win_predictions_elo_diff.item(),tie_predictions_elo_diff.item(),loss_predictions_elo_diff.item()]
    

    # ----- Select the team -----
    match_data = create_match(team1,team2)
    result = ml_simulate_match(match_data)

    form_adv = match_data["rolling_points_diff"][0]
    form_adv_probs = get_points_probabilities(form_adv)
    curr_eff_adv = match_data["rolling_score_diff"][0]
    curr_eff_adv_probs = get_score_probabilities(curr_eff_adv)
    qual_adv = int(match_data["elo_diff"][0])
    qual_adv_probs = get_elo_probabilities(qual_adv)

    quotes = [(form_adv_probs[0]*0.25)+(curr_eff_adv_probs[0]*0.25)+(qual_adv_probs[0]*0.5),(form_adv_probs[1]*0.25)+(curr_eff_adv_probs[1]*0.25)+(qual_adv_probs[1]*0.5),(form_adv_probs[2]*0.25)+(curr_eff_adv_probs[2]*0.25)+(qual_adv_probs[2]*0.5)]
    
    st.write("###")
    st.write(f"""<div style="text-align: center;"><h2 style="text-align: center;">{team1} {result[0][0]} :  {team2} {result[0][1]}</h1></div>""", unsafe_allow_html=True)
    st.write(f"""<div style="text-align: center;"><p style="text-align: center;">AI Prediction</p></div>""", unsafe_allow_html=True)

    st.write("###")
    st.write(f"""<div style="text-align: center;"><h3 style="text-align: center;"> 1: {round(1/quotes[0],2)} | X: {round(1/quotes[1],2)} | 2: {round(1/quotes[2],2)}  </h1></div>""", unsafe_allow_html=True)
    st.write(f"""<div style="text-align: center;"><p style="text-align: center;">Fair Odds</p></div>""", unsafe_allow_html=True)
    st.write("###")
    
    col_matchinfo = st.columns([3, 3, 3])



    col_matchinfo[0].write("*Current Form Advantage*")
    col_matchinfo[0].write(f"### {form_adv}")
    col_matchinfo[0].write(f"1: {round((form_adv_probs[0])*100,2)}% | X: {round((form_adv_probs[1])*100,2)}% | 2: {round((form_adv_probs[2])*100,2)}%")
    col_matchinfo[1].write("*Current Efficience Advantage*")
    col_matchinfo[1].write(f"### {curr_eff_adv}")
    col_matchinfo[1].write(f"1: {round((curr_eff_adv_probs[0])*100,2)}% | X: {round((curr_eff_adv_probs[1])*100,2)}% | 2: {round((curr_eff_adv_probs[2])*100,2)}%")
    col_matchinfo[2].write("*Quality Advantage*")
    col_matchinfo[2].write(f"### {qual_adv}")
    col_matchinfo[2].write(f"1: {round((qual_adv_probs[0])*100,2)}% | X: {round((qual_adv_probs[1])*100,2)}% | 2: {round((qual_adv_probs[2])*100,2)}%")

    

# streamlit run Dashboard.py