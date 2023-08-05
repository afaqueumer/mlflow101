# Import Libraries
import streamlit as st  
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import subprocess
import os
import webbrowser

# Configure Page
st.set_page_config(
    page_title="Spam Filter",
    page_icon="🤖", 
    layout="centered",
    initial_sidebar_state="expanded") 

# load feature extracted data
df = pd.read_csv("data.csv")

# HELPER FUNCTIONS

# A bsic text processing function with options for with/without stop words or
# stemming / lemmatizing
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text.lower())
    # filtered_words = [word for word in words if word.isalpha()]
    # filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    filtered_words = [stemmer.stem(word) for word in words if word.isalpha() and word not in stop_words]
    # filtered_words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(filtered_words)

# Train the model
def train_model(exp_name, df, n, c, d): 
    
    df['processed_message'] = df.message.apply(preprocess_text)
    # Split the data into features (X) and labels (y)
    x = df['processed_message']
    y = df['label']
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # Create or Select Experiment 
    experiment = mlflow.set_experiment(exp_name)    
    with mlflow.start_run(experiment_id=experiment.experiment_id):          
        # Create a Vectorizer to convert text data to numerical features: BoW / TF-IDF 
        # vectorizer = CountVectorizer()
        vectorizer = TfidfVectorizer()
        x_train_vectorized = vectorizer.fit_transform(x_train)          
        x_test_vectorized = vectorizer.transform(x_test)          
        rf_classifier = RandomForestClassifier(n_estimators=n, criterion=c, max_depth=d)
        rf_classifier.fit(x_train_vectorized, y_train)
        # Make predictions on the training & test set
        y_train_pred = rf_classifier.predict(x_train_vectorized)
        y_test_pred = rf_classifier.predict(x_test_vectorized)
        # Evaluate the model
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred, pos_label='spam')        
        # Log Parameters & Metrics
        mlflow.log_params({"n_estimators":n, "Criterion": c, "Maximum Depth": d})        
        mlflow.log_metrics({"Training Accuracy": train_acc, "Test Accuracy": test_acc, "F1 Score": f1})
        # Log Model & Vectorizer
        mlflow.sklearn.log_model(rf_classifier, "model")
        mlflow.sklearn.log_model(vectorizer, "vectorizer") 
    return train_acc, test_acc

# Function for opening MLFlow UI directly from Streamlit
def open_mlflow_ui():
    # Start the MLflow tracking server as a subprocess
    cmd = "mlflow ui --port 5000"
    subprocess.Popen(cmd, shell=True)
def open_browser(url):
    webbrowser.open_new_tab(url)
    
# STREAMLIT UI  
 
# Sidebar for hyperparameter tuning
st.sidebar.title("Tune Hyper Params ⚙️")
n = st.sidebar.slider('N-Estimators',min_value=1, max_value=200, step=2, value=10)
d = st.sidebar.slider('Max Depth', min_value=1, max_value=20, step=2, value=2)
c = st.sidebar.selectbox('Criterion', ['gini', 'entropy', 'log_loss'], index=1)

# Launch Mlflow from Streamlit
st.sidebar.title("Mlflow Tracking 🔎")    
if st.sidebar.button("Launch 🚀"):
    open_mlflow_ui()
    st.sidebar.success("MLflow Server is Live! http://localhost:5000")
    open_browser("http://localhost:5000")

# Main Page Content
st.title("Spam Classifier Trainer 🤖")
exp_type = st.radio("Select Experiment Type", ['New Experiment', 'Existing Experiment'], horizontal=True)
if exp_type == 'New Experiment':
    exp_name = st.text_input("Enter the name for New Experiment")
else:
    try:
        if os.path.exists('./mlruns'):
            exps = [i.name for i in mlflow.search_experiments()]
            exp_name = st.selectbox("Select Experiment", exps)
        else:
            st.warning("🚨 No Previous Experiments Found! Set New Experiment ⬆️")            
    except:
        st.warning("🚨 No Previous Experiments Found! Set New Experiment ⬆️")

# Training the model starts from here    
if st.button("Train ⚙️"):
    with st.spinner('Feeding the data--->🧠'):
        tr_a, ts_a = train_model(exp_name, df, n, c, d)
    st.success('Trained!') 
    st.write(f"Training Accuracy Achieved: {tr_a:.3f}")      