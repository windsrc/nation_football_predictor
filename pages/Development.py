import streamlit as st

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

# Title of the application
st.title("Development of a National Football Predictor")

# Abstract section
st.header("Abstract")
st.write(
    """
    This document outlines the development of a multi-output machine learning model designed to predict 
    the probabilities of wins, ties, and losses in football matches. The model leverages historical match data, 
    computes advanced rolling statistics, and applies Elo rating systems to quantify team strengths. 
    Sigmoid curve fitting is employed to model outcome probabilities, and the final model has been pruned 
    to ensure it is small enough for efficient deployment on platforms like GitHub.
    """
)

# Introduction section
st.header("1. Introduction")
st.write(
    """
    Predicting football match outcomes is a complex task that requires sophisticated statistical modeling 
    and feature engineering. The objective of this project was to develop an application capable of providing 
    accurate predictions for match results based on historical data, rolling statistics, and team strength indicators.
    """
)

# Data Import and Preprocessing section
st.header("2. Data Import and Preprocessing")
st.write(
    """
    - **Library Imports:** The project began with the importation of essential Python libraries, including 
        `pandas`, `numpy`, `matplotlib`, and `seaborn`.
    - **Data Loading:** Match results were imported from a CSV file into a pandas DataFrame, with the date column 
        converted to a `datetime` format for accurate chronological processing.
    """
)

# Score Calculation section
st.header("3. Score Calculation")
st.write(
    """
    - **Function Creation:** A function was developed to calculate points and net goals based on match results. 
        This function was applied to create a DataFrame containing individual records for both home and away teams.
    - **Data Sorting:** The resulting DataFrame was sorted by date to maintain the correct chronological order, 
        ensuring the integrity of rolling statistics.
    """
)

# Rolling Statistics section
st.header("4. Rolling Statistics")
st.write(
    """
    - **Performance Tracking:** Rolling sums of goals and points over the last 10 games for each team were calculated 
        to track performance trends over time.
    - **Opponent Statistics:** Similar rolling statistics were computed for each team's opponents to provide a comparative analysis.
    """
)


# Elo Rating Calculation section
st.header("5. Elo Rating Calculation")
st.write(
    """
    - **Elo Function Development:** A function was created to compute Elo ratings, which are updated after each match 
        based on the results. Elo ratings offer a quantifiable measure of team strength relative to their opponents.
    - **Elo Integration:** These Elo ratings were integrated into the main DataFrame, allowing for comparison across matches.
    """
)
st.markdown(
        """
        ```latex
        # Calculate expected scores
        expected_team1 = 1 / (1 + 10 ** ((elo_team2 - elo_team1) / 600))
        expected_team2 = 1 / (1 + 10 ** ((elo_team1 - elo_team2) / 600))

        # Update Elo ratings
        new_elo_team1 = elo_team1 + k * (team1_result - expected_team1)
        new_elo_team2 = elo_team2 + k * (team2_result - expected_team2)
        ```
        """
    )

# Feature Engineering section
st.header("6. Feature Engineering")
st.write(
    """
    - **Feature Computation:** Differences between the rolling statistics (points, net score) and Elo ratings for teams 
        and their opponents were computed. These features serve as the primary predictors of match outcomes.
    """
)

# Statistical Analysis section
st.header("7. Statistical Analysis")
st.write(
    """
    - **Result Aggregation:** Match results were aggregated by rolling points, score difference, and Elo difference to 
        calculate probabilities for wins, ties, and losses.
    - **Data Visualization:** Bar plots were generated to visualize these probabilities, enhancing the understanding of 
        the relationship between the features and match outcomes.
    """
)
st.image("chart_point_diff.png", width=700)
st.write("*Chart: Rolling Points Difference of the last 10 matches with probabilites*")
st.write("###")
st.image("chart_score_diff.png", width=700)
st.write("*Chart: Rolling Score Difference of the last 10 matches with probabilites*")
st.write("###")
st.image("chart_elo_diff.png", width=700)
st.write("*Chart: ELO Score Difference with probabilites*")
st.write("###")


# Sigmoid Curve Fitting section
st.header("8. Sigmoid Curve Fitting")
st.write(
    """
    - **Model Application:** Sigmoid functions were applied to model the probabilities of wins, ties, and losses based on 
        rolling points, score differences, and Elo differences.
    - **Curve Fitting:** The sigmoid curves were fitted to the aggregated data, enabling predictions based on these features.
    """
)
st.image("sigmoid_curve.png", width=700)
st.write("*Chart: Fitted Sigmoid Curve for prediciting probabilities (Rolling Score Difference - Wins)*")

# Probability Prediction section
st.header("9. Probability Prediction")
st.write(
    """
    - **Prediction Functions:** Functions were developed to predict the probabilities of wins, ties, and losses given new data 
        on rolling points and score differences.
    - **Model Pruning:** To ensure that the model is lightweight enough for deployment on GitHub, pruning techniques were applied, 
        optimizing the model's size without compromising its predictive accuracy.
    """
)

# Conclusion section
st.header("10. Conclusion")
st.write(
    """
    The application successfully integrates historical match data, calculates advanced rolling and comparative statistics, 
    models outcome probabilities using curve fitting, and provides insightful visualizations. The multi-output model is now 
    ready for deployment, offering robust predictions for football match outcomes.
    """)