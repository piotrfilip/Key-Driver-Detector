# Key Driver Detector

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyCaret](https://img.shields.io/badge/PyCaret-000000?style=for-the-badge&logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)

Key Driver Detector is an interactive end-to-end Machine Learning web application built with Streamlit. It allows users to upload raw datasets, automatically trains predictive models to identify the most critical features (key drivers) impacting a target variable, and leverages OpenAI's GPT-4o model to provide actionable business insights.

## Key Features

* **Smart Data Handling:** Upload CSV files with automatic delimiter detection using Python's `csv.Sniffer`.
* **Automated Preprocessing:** Automatically handles missing records, drops user-specified columns, and converts time-based string columns into computational seconds.
* **AutoML Integration:** Dynamically classifies the problem as Regression or Classification based on the target column's data type and trains multiple algorithms (Linear Regression/Logistic Regression, Decision Trees, Random Forest, LightGBM) using PyCaret.
* **Permutation Feature Importance:** Calculates and visualizes feature impact using `sklearn.inspection.permutation_importance` and Plotly interactive bar charts.
* **AI-Powered Business Insights:** Integrates the OpenAI API (GPT-4o) via the `instructor` library and `pydantic` schemas to generate structured, strictly typed analytical reports and actionable optimization tips based on model metrics.

## Tech Stack

* **Frontend & UI:** Streamlit
* **Data Manipulation:** Pandas
* **Machine Learning:** PyCaret, Scikit-learn
* **Data Visualization:** Plotly Express
* **GenAI / LLM:** OpenAI API, Instructor, Pydantic

## Installation & Local Setup

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/key-driver-detector.git](https://github.com/yourusername/key-driver-detector.git)
   cd key-driver-detector
   ```

2. Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```

3. Run the Streamlit application
 ```bash
  streamlit run app.py
  ```

## Author

Piotr Filipowski
LinkedIn (www.linkedin.com/in/piotrfilipowski)
