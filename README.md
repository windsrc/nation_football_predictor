## National Football Team Prediction Tool - Streamlit App README

This README file provides an overview of the National Football Team Prediction Tool, a Streamlit app designed to predict football match outcomes using a machine learning model.

**Author:** Christopher Windsor ([GitHub](https://github.com/windsrc))

**Description**

This interactive app allows users to select two national teams from the EURO 2024 and predicts the likelihood of a win, tie, or loss for each team. The prediction is based on historical performance data and a trained machine learning model. Additionally, the app generates fair betting odds and visualizes the predicted score distributions. 

**Features**

* Select two national teams from the EURO 2024
* View the predicted match outcome (win, tie, loss) for each team
* Access fair betting odds based on the predictions
* Explore the predicted score distributions for each team (Current Form Advantage, Current Efficiency Advantage, Quality Advantage)

**Requirements**

* Python 3.x
* Streamlit library
* Pandas library
* NumPy library
* scikit-learn library (for the pre-trained model)
* Pickle library (for loading the model)
* Matplotlib library (optional, for visualizations not currently included)
* Seaborn library (optional, for visualizations not currently included)

**Installation**

1. Create a virtual environment (recommended):
    ```bash
    python3 -m venv national_football_env
    source national_football_env/bin/activate  # Activate the virtual environment
    ```
2. Install required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. Download the pre-trained model file (`trained_multi_output_model.pkl`) and place it in the `model` directory.
4. Download the historical results data (`modified_results.csv`) and place it in the `data` directory.

**Running the App**

1. Navigate to the directory containing the `Dashboard.py` script.
2. Run the script using Streamlit:
    ```bash
    streamlit run Dashboard.py
    ```

**Code Structure**

The `Dashboard.py` script utilizes Streamlit components to create the interactive app. Here's a breakdown of the key functionalities:

* **Page Configuration:** Sets the title and favicon of the app.
* **Left Menu:** Provides a brief description of the app and credits the author.
* **Team Selection:** Allows users to choose two teams from a list using a multi-select dropdown.
* **Calculations:**
    * Loads the pre-trained model from the `model` directory.
    * Retrieves the latest team data based on the selected teams.
    * Calculates various performance metrics (rolling points, net score, Elo rating) for both teams.
    * Uses the model to predict the match outcome and calculates probabilities for win, tie, and loss.
    * Generates fair betting odds based on the predicted probabilities.
* **Results Display:**
    * Shows the predicted match score and AI prediction label.
    * Displays the predicted win, tie, and loss probabilities along with fair betting odds.
    * Visualizes the predicted score distributions for each advantage factor (Current Form, Current Efficiency, Quality).

**Note:**

* The visualizations for predicted score distributions (using Matplotlib or Seaborn) are currently commented out in the code. You can uncomment them and install the respective libraries if you want to include these visualizations.


This README provides a comprehensive overview of the National Football Team Prediction Tool and its functionalities. Feel free to explore the code and customize it further to fit your specific needs.