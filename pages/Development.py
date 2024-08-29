# Title of the application
st.title("Development of a Multi-Output Predictive Model for Football Match Outcomes")

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

# Sigmoid Curve Fitting section
st.header("8. Sigmoid Curve Fitting")
st.write(
    """
    - **Model Application:** Sigmoid functions were applied to model the probabilities of wins, ties, and losses based on 
        rolling points, score differences, and Elo differences.
    - **Curve Fitting:** The sigmoid curves were fitted to the aggregated data, enabling predictions based on these features.
    """
)

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