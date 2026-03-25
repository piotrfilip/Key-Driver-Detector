import streamlit as st
import pandas as pd
import csv
import pycaret.regression as reg
import pycaret.classification as clf
import plotly.express as px
import instructor
import base64
import time
from openai import OpenAI
from sklearn.inspection import permutation_importance
from pydantic import BaseModel, Field

# Function to create an OpenAI client using the API key stored in session state

def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])

# Function to detect the delimiter in a CSV file

def separator_selection(uploaded_file):
    content = uploaded_file.read(6048).decode("utf-8")
    uploaded_file.seek(0)
    dialect = csv.Sniffer().sniff(content)
    return dialect.delimiter

# Function to classify the problem type based on the target column's data type and number of distinct values

def classify_problem_type(df, target_column):
    distinct_counts = df[target_column].nunique()
    if pd.api.types.is_numeric_dtype(df[target_column]) and distinct_counts >= 5:
        return "regression"
    else:
        return "classification"
    
# Functions to prepare the dataset: dropping empty records, removing specified columns, and converting time columns to seconds

def drop_empty_records(df, target_column):
    df = df.dropna(subset=[target_column])
    return df

def drop_columns(df, delete_columns):
    if delete_columns:
        df = df.drop(columns=delete_columns)
    return df

def convert_time_to_seconds(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                converted_col = pd.to_timedelta(df[col], errors='raise')
                df[col] = converted_col.dt.total_seconds().astype('Int64')
            except (ValueError, TypeError):
                pass  
    return df

def convert_target_to_numeric(df, target_column):

    if df[target_column].dtype == 'object':
        df[target_column] = df[target_column].astype(str).str.replace(',', '', regex=True).str.strip()
        df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
        
    return df

# Function to train a PyCaret model and return the best model along with training data

def load_pycaret_model(problem_type, dataframe, target_column):

    if problem_type == "regression":

        exp = reg.setup(
            data=dataframe, 
            target=target_column, 
            verbose=False, 
            session_id=67,  
            remove_outliers=True,
            normalize=True
        )
        best_model = exp.compare_models(verbose=False, fold=3, include=['lr', 'dt', 'rf', 'lightgbm'])
        try:
            X_train = reg.get_config('X_train_transformed')
            y_train = reg.get_config('y_train_transformed')
        except (KeyError, ValueError):
            X_train = reg.get_config('X_train')
            y_train = reg.get_config('y_train')
        
        return best_model, X_train, y_train
             
    elif problem_type == "classification":
        
        dataframe = dataframe.dropna(subset=[target_column])

        try:
            exp = clf.setup(
                data=dataframe, 
                target=target_column, 
                verbose=False, 
                session_id=67,
                remove_outliers=True,    
                fix_imbalance=True
            )
            best_model = exp.compare_models(verbose=False, fold=3, include=['lr', 'dt', 'rf', 'lightgbm'])
            try:
                X_train = clf.get_config('X_train_transformed')
                y_train = clf.get_config('y_train_transformed')
            except (KeyError, ValueError):
                X_train = clf.get_config('X_train')
                y_train = clf.get_config('y_train')
            
            return best_model, X_train, y_train

        except ValueError as e:
            if "least populated class" in str(e) or "too few" in str(e):
                st.error("Data imbalance issue detected: One or more classes in the target variable have too few samples after outlier removal. Please consider adjusting the outlier removal settings or using a different target column.")
            else:
                st.error(f"Error during model training: {e}")
            return None, None, None
            
        except Exception as e:
            st.error(f"Unexpected error during model training: {e}")
            return None, None, None
    
    elif problem_type not in ["reg", "clf"]:
        st.error("Unsupported problem type. Please select a valid target column.")
        return None, None, None
    
# Function to calculate and display permutation feature importance using Plotly

def plot_permutation_importance(model, X_train, y_train):
    result = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=67)
    perm_importances_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": result.importances_mean
    }).sort_values(by="importance", ascending=True)
    
    fig = px.bar(
        perm_importances_df, 
        x="importance", 
        y="feature",
        orientation="h"
    )
    
    dynamic_height = max(400, len(perm_importances_df) * 25)

    fig.update_layout(
        title="Permutation Importances of Features",
        xaxis_title="Mean Importance",
        yaxis_title="Feature",
        height=dynamic_height,
        margin=dict(t=60, b=40, l=150, r=20)
    )
    return fig

# Function to generate a description of the dataset using OpenAI

class AnalizeRaport(BaseModel):
    feature_importance_desc: str = Field(..., description="A cohesive, analytical summary focusing ONLY on the top key drivers impacting the model. Ignores features with near-zero importance.")
    optimization_tips: list[str] = Field(..., description="List of actionable tips for optimizing the dataset, feature engineering, and improving model performance.")

def generate_dataset_description(df, fig):
    openai_client = get_openai_client()
    instructor_client = instructor.from_openai(openai_client)
    
    features = fig.data[0].y
    importances = fig.data[0].x
    
    importance_data = "\n".join([f"- {f}: {i:.4f}" for f, i in zip(features, importances)])
    
    df_info = f"Data set has {df.shape[0]} rows and {df.shape[1]} columns. Available columns are: {', '.join(df.columns)}."
    
    raport = instructor_client.chat.completions.create(
        model="gpt-4o",
        response_model=AnalizeRaport,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert Data Scientist and Business Analyst. Analyze the dataset information and the permutation feature importance scores. "
                    "Provide a comprehensive and insightful summary of the results. Focus ONLY on the top key drivers—the features that have the most significant impact on the target variable. "
                    "Explain their potential real-world or business context if possible. "
                    "Do NOT list every single feature. Group or completely ignore features with near-zero or negative importance. "
                    "Provide your expert conclusion and actionable optimization tips to improve model performance or data quality."         
                )
            },
            {
                "role": "user",
                "content": (
                    f"Here is the basic information about the dataset:\n{df_info}\n\n"
                    f"Here are the Permutation Feature Importances calculated by our machine learning model:\n{importance_data}\n\n"
                    "Please analyze this data and provide your expert summary."
                )
            }
        ]
    )
    
    return raport

# Function to reset the app state when uploading a new dataset or changing the target column
def reset_app_state():
    if 'prepared_df' in st.session_state:
        st.session_state.prepared_df = None
    if 'feature_importance_plot' in st.session_state:
        del st.session_state['feature_importance_plot']
    if 'analysis_complete' in st.session_state:
        st.session_state['analysis_complete'] = False
    if 'start_ai_analysis' in st.session_state:
        st.session_state['start_ai_analysis'] = False
    if 'target_col_key' in st.session_state:
        st.session_state['target_col_key'] = None
    if 'delete_cols_key' in st.session_state:
        st.session_state['delete_cols_key'] = []
    if 'ai_raport' in st.session_state:
        del st.session_state['ai_raport']

# ==========================================
## MAIN APP
# ==========================================

st.set_page_config(page_title="Key Driver Detector", layout="wide")

st.title("Key Driver Detector 🔎🔑")

# ==========================================
# 1. INITIALIZATION OF VARIABLES AND SESSION STATE
# ==========================================
df = pd.DataFrame()
prep_button_disabled = True
run_button_disabled = True

if 'prepared_df' not in st.session_state:
    st.session_state.prepared_df = None
if "start_ai_analysis" not in st.session_state:
    st.session_state["start_ai_analysis"] = False

if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# ==========================================
# 2. SIDEBAR - UPLOAD, SETTINGS, INSTRUCTIONS
# ==========================================
with st.sidebar:

    with st.popover("How does it work? 💡",use_container_width=True):
        st.markdown(f"Hi 👋 \n\n There is quick start instructions to help you get started with the Key Driver Detector app:")
        st.write("1. Upload your dataset in CSV format. The app will automatically detect the separator used in your file.")
        st.write("2. Select the target column you want to analyze. You can also choose to exclude certain columns from the analysis if you think they are not relevant.")
        st.write("3. Click 'Prepare Data' to clean and preprocess your dataset. This includes dropping empty records, removing specified columns, and converting time columns to seconds.")
        st.write("4. After preparing the data, click 'Run Key Driver Analysis' to train a model and calculate the permutation importance of each feature.")
        st.write("5. Once the analysis is complete, you can generate a detailed dataset description and optimization tips using OpenAI. Just provide your OpenAI API key when prompted.")
        st.write("6. If you want to start over or upload a new dataset, simply click the 'Reset App State' button at the bottom of the sidebar.")
        st.markdown("Feel free to experiment with different datasets and target columns to discover the key drivers in your data!")

    st.divider()
    
    st.header("Step 1: Data Upload 📥")
    
    uploaded_file = st.file_uploader(type=["csv"], label="Upload your dataset in CSV", on_change=reset_app_state)
    
    if uploaded_file is not None:
        try:
            separator = separator_selection(uploaded_file)
            df = pd.read_csv(uploaded_file, sep=separator)
        except Exception as e:
            st.error(f"Error processing the uploaded file: {e}")

    st.divider()

    st.header("Step 2: Column Selection 📊")
    columns_list = df.columns.tolist() if not df.empty else []
    
    target_column = st.selectbox(
        "Select target column:", 
        columns_list if columns_list else [],
        index=None,
        disabled=st.session_state.analysis_complete,
        key="target_col_key"
    )
    
    delete_columns = st.multiselect(
        "Select columns to exclude (optional):", 
        columns_list,
        key="delete_cols_key"
    )

    if target_column is None or target_column in delete_columns:
        prep_button_disabled = True
    else:
        prep_button_disabled = False

    st.divider()

    # c) Button 1: Prepare Data
    st.header("Step 3: Prepare Data 🛠️")
    if st.button("1. Prepare Data", disabled=prep_button_disabled):
        with st.spinner("Preparing data..."):

            df_temp = drop_columns(df, delete_columns)      
            df_temp = convert_time_to_seconds(df_temp)
            df_temp = convert_target_to_numeric(df_temp, target_column)
            df_temp = drop_empty_records(df_temp, target_column)
            
            st.session_state.prepared_df = df_temp
            st.toast("Data prepared!", icon="✅")

    st.divider()

    # d) Button 2: Run Analysis
    st.header("Step 4: Run Key Driver Analysis 🚀")

    run_button_disabled = True
    if st.session_state.prepared_df is not None and target_column:
        run_button_disabled = False
        
    if st.button("2. Run Key Driver Analysis", disabled=run_button_disabled):
        with st.spinner("Analyzing..."):
            df_ready = st.session_state.prepared_df
            problem_type = classify_problem_type(df_ready, target_column)
            
            model_plot = load_pycaret_model(problem_type, df_ready, target_column)
            if model_plot[0]: 
                st.toast("Key Driver Analysis completed!", icon="✅")
                fig = plot_permutation_importance(model_plot[0], X_train=model_plot[1], y_train=model_plot[2])
                st.session_state["feature_importance_plot"] = fig 
                st.session_state.analysis_complete = True
                st.rerun()

    st.divider()

    # e) Button 3: OpenAI Analysis
    st.header("Step 5: AI Dataset Analysis 🤖")
    
    ai_button_disabled = "feature_importance_plot" not in st.session_state
    
    if st.button("Generate Dataset Description", disabled=ai_button_disabled):
        st.session_state["start_ai_analysis"] = True
            
    if st.session_state.get("start_ai_analysis"):
        if not st.session_state.get("openai_api_key"):
            st.info("Provide your OpenAI API key.")
            api_key_input = st.text_input("OpenAI API Key", type="password")
            if api_key_input:
                with st.spinner("Verifying API key..."):
                    try:
                        test_client = OpenAI(api_key=api_key_input)
                        test_client.models.list()
                        st.session_state["openai_api_key"] = api_key_input
                        st.success("API key verified!")
                        time.sleep(2)
                        st.rerun()
                    except Exception as e:
                        error_msg = str(e)
                        if "401" in error_msg or "invalid_api_key" in error_msg:
                            st.error("🔑 Incorrect API key provided.")
                        else:
                            st.error(f"Error: {e}")
        else:
            with st.spinner("OpenAI is analyzing the dataset..."):
                try:
                    raport = generate_dataset_description(
                        df=st.session_state.prepared_df, 
                        fig=st.session_state["feature_importance_plot"]
                    )
                    st.session_state["ai_raport"] = raport
                    st.session_state["start_ai_analysis"] = False 
                except Exception as e:
                    st.error(f"Error generating AI description: {e}")
                    if "401" in str(e):
                        st.session_state["openai_api_key"] = None

    st.divider() 
    st.markdown("### 🔄 Reset")
    st.write("Would you like to change target column? Or maybe upload a new dataset? Click the button below to reset the app state and start fresh.")
    
    st.button(
        "Reset App State", 
        on_click=reset_app_state, 
        type="primary", 
        use_container_width=True 
    )

# ==========================================
# 3. Main content
# ==========================================

if df.empty:
    st.info("👈 Please upload a dataset in CSV format to get started.")

is_target_valid = target_column is not None and target_column not in delete_columns

# A) Raw dataset sample
if not df.empty:
    is_raw_expanded = st.session_state.prepared_df is None or not is_target_valid
    
    with st.expander("Raw Dataset Sample", icon="📂", expanded=is_raw_expanded):
        st.caption("💡 This section shows a sample of the raw dataset immediately after uploading the CSV file. It allows you to quickly inspect the data before any cleaning or preprocessing steps are applied.")
        st.dataframe(df.sample(min(10, len(df))))
        
if target_column is None and not df.empty:
    st.info("👈 Please select a target column in the sidebar to continue.")
elif target_column in delete_columns:
    st.warning("👈 Target column cannot be excluded. Please adjust the sidebar settings.")

if st.session_state.prepared_df is not None and is_target_valid:
    
    # B) Prepared dataset
    with st.expander("Prepared Dataset Sample", icon="🛠️", expanded=True):
        st.caption("💡 This section displays a sample of the dataset after it has been prepared. The preparation steps include dropping empty records, removing specified columns, and converting time columns to seconds. This allows you to see how the data looks before running the key driver analysis.")
        st.dataframe(st.session_state.prepared_df.sample(min(10, len(st.session_state.prepared_df))))
        
        detected_type = classify_problem_type(st.session_state.prepared_df, target_column)
        st.metric(label="Detected problem type:", value=detected_type, border=True)

    if "feature_importance_plot" not in st.session_state:
        st.info("👈 Please run the key driver analysis to see the feature importance plot and generate the dataset description.")

    # C) Feature Importance Plot
    if "feature_importance_plot" in st.session_state:
        with st.expander("Feature Importance Plot", icon="📊", expanded=True):
            st.caption("💡 This plot shows the permutation importance of each feature in the dataset based on the trained model. The importance values indicate how much each feature contributes to the model's predictions. Higher importance means that the feature has a greater impact on the target variable.")
            st.plotly_chart(
                st.session_state["feature_importance_plot"], 
                use_container_width=True
            )

    if "ai_raport" not in st.session_state and "feature_importance_plot" in st.session_state:
        st.info("👈 You can generate a detailed dataset description and optimization tips using OpenAI. Just provide your OpenAI API key when prompted.")

    # D) AI Raport
    if "ai_raport" in st.session_state:
        with st.expander("Dataset Analysis Results", icon="🤖", expanded=True):
            st.caption("💡 This section provides a detailed analysis of the dataset based on the feature importance plot and the dataset information. The description includes insights into the importance of each feature and actionable tips for optimizing the dataset and improving model performance, all generated by OpenAI's language model.")
            st.write("### 📄 Dataset Description")
            raport = st.session_state["ai_raport"]
            st.write(raport.feature_importance_desc)
            
            st.write("### 💡 Optimization Tips")
            for tip in raport.optimization_tips:
                st.markdown(f"- {tip}")
