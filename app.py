import os
import json
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
        page_title="PHA Classification",
        page_icon="☄️",
        layout="centered"
    )

st.title("Planetary Defense Dashboard☄️")

with st.sidebar:
    selected = option_menu("PLANETARY DEFENSE DASHBOARD", ['PHA Classification','Diameter Prediction', 'Asteroid Clustering'], 
        icons=['asteroid','bar-chart-line','diagram-3'], menu_icon="cast", default_index=1) 

if selected == "PHA Classification":
    st.subheader("Potentially Hazardous Asteroid Classification")
    
    MODEL_NAME = "PHA_RandomForest_Classifier"
    MODEL_VERSION = "1"

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    @st.cache_resource
    def load_model():
        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        return mlflow.sklearn.load_model(model_uri)


    @st.cache_data
    def load_feature_config():
        with open("feature_config.json", "r") as f:
            return json.load(f)


    try:
        model = load_model()
        feature_config = load_feature_config()
    except Exception as e:
        st.error(f"Unable to load the model or feature configuration: {e}")
        st.stop()

    st.write(
            "Enter asteroid feature values."
        )
    
    input_values = {}

    for feature in feature_config:
        input_values[feature["name"]] = st.number_input(
            label=feature["name"],
            value=float(feature["default"]),
            format="%.6f"
        )


    if st.button("Predict asteroid classification"):
        input_df = pd.DataFrame(
            [input_values],
            columns=[f["name"] for f in feature_config]
        )

        try:
            prediction = int(model.predict(input_df)[0])
            probabilities = model.predict_proba(input_df)[0]

            st.subheader("Prediction result")

            if prediction == 1:
                st.warning("Model classification: Potentially Hazardous Asteroid (PHA)")
            else:
                st.success("Model classification: Non-PHA")

            st.write("Predicted class:", prediction)

            st.write("Class probabilities:")
            st.write({
                "Non-PHA": float(probabilities[0]),
                "PHA": float(probabilities[1])
            })

            st.caption(
                "This is a machine-learning classification result, "
                "not an authoritative assessment of collision risk."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

if selected == "Diameter Prediction":
    st.subheader("Asteroid Diameter Estimation")
    
    MODEL_NAME = "Asteroid_Diameter_Regressor"
    MODEL_VERSION = "1"


    @st.cache_resource
    def load_model():
        return mlflow.sklearn.load_model(
            f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        )


    @st.cache_data
    def load_feature_config():
        with open("feature_config.json", "r") as file:
            return json.load(file)


    try:
        model = load_model()
        feature_config = load_feature_config()
    except Exception as exc:
        st.error(f"Unable to load model or configuration: {exc}")
        st.stop()

    st.write(
        "Enter asteroid features to estimate its diameter ")
    
    input_values = {}

    for feature in feature_config:
        input_values[feature["name"]] = st.number_input(
            feature["name"],
            value=float(feature["default"]),
            format="%.6f"
        )


    if st.button("Predict Diameter"):

        input_df = pd.DataFrame(
            [input_values],
            columns=[feature["name"] for feature in feature_config]
        )

        prediction = float(model.predict(input_df)[0])

        st.success(f"Predicted diameter: {prediction:.4f}")

        st.caption(
            "Check the dataset's target units before interpreting "
            "the prediction. This is a model estimate, not a direct "
            "asteroid diameter measurement."
        )

if selected == "Asteroid Clustering":
    st.subheader("Asteroid Clustering")

    model_name = "Asteroid_KMeans_Model"
    model_version = "1"

    model_uri = f"models:/{model_name}/{model_version}"

    @st.cache_resource
    def load_model():
        return mlflow.sklearn.load_model(model_uri)

    model = load_model()

    feature_names = model.feature_names_in_

    input_values = {}

    for feature in feature_names:
        input_values[feature] = st.number_input(
            feature,
            value=0.0,
            format="%.6f"
        )

    if st.button("Predict Cluster"):

        input_df = pd.DataFrame([input_values])

        cluster = model.predict(input_df)[0]

        st.success(f"Predicted cluster: {cluster}")