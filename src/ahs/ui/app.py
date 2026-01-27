import os
import sys

# Add project root to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import requests
import streamlit as st
import numpy as np
import cv2
from src.ahs.utils.visualize_img import visualize_prediction_data
import pandas as pd
from dotenv import load_dotenv

# Load env var if not already
load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8080")

st.set_page_config(
    page_title="AHS",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Medical Theme
st.markdown("""
    <style>
    .reportview-container {
        background: #f8fafc;
    }
    .sidebar .sidebar-content {
        background: #0f172a;
        color: white;
    }
    h1, h2, h3 {
        color: #0077b6;
        font-family: 'sans-serif';
    }
    .stButton>button {
        background-color: #0077b6;
        color: white;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("AHS")
    st.success("System Online")
    
    page = st.radio("Navigation", ["Analysis", "History"])
    
    st.markdown("---")
    st.markdown("**User:** Hiep")
    st.markdown("**Role:** Hematologist")

if page == "Analysis":
    st.title("Blood Cell Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Upload Sample")
        uploaded_file = st.file_uploader("Choose a microscopic image...", type=["jpg", "png", "jpeg"])
        
        confidence = st.slider("Detection Sensitivity", 0.1, 0.9, 0.5)
        
        if uploaded_file is not None:
            # Display preview
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, 1)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            st.image(image_rgb, caption="Original Image", width="stretch")
            
            if st.button("Run Analysis", type="primary"):
                with st.spinner("Analyzing sample..."):
                    # Reset pointer to send to API
                    uploaded_file.seek(0)
                    files = {"file": uploaded_file}
                    try:
                        resp = requests.post(f"{API_URL}/predict", params={"score_thresh": confidence}, files=files)
                        if resp.status_code == 200:
                            data = resp.json()
                            
                            # Visualize
                            # API returns boxes, labels, scores.
                            # Need to draw on the original image (numpy)
                            # visualize_prediction_data expects BGR/RGB? The docstring says it draws in BGR (opencv default).
                            # If we pass RGB (which st.image expects), we might get color inversion if util treats it as BGR.
                            # visualize_prediction_data signature: image_np, boxes, labels...
                            # It copies image_np. It uses id2color which is defined as (0, 0, 255) for RBC.
                            # Usually cv2 uses BGR. So (0,0,255) is Red in BGR.
                            # If we pass RGB image where pixels are R,G,B -> Red is (255,0,0).
                            # If we pass RGB to cv2.rectangle with color (0,0,255), it draws Blue on an RGB image.
                            # So we should pass BGR image to the function, get BGR result, then convert to RGB for Streamlit.
                            
                            img_result_bgr = visualize_prediction_data(
                                image_bgr, 
                                data["boxes"], 
                                data["labels"], 
                                data["scores"]
                            )
                            img_result_rgb = cv2.cvtColor(img_result_bgr, cv2.COLOR_BGR2RGB)
                            
                            st.session_state["last_result"] = img_result_rgb
                            st.session_state["last_meta"] = data["meta"]
                            
                        else:
                            st.error(f"Analysis failed: {resp.text}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")

    with col2:
        st.subheader("Results")
        if "last_result" in st.session_state:
            st.image(st.session_state["last_result"], caption="Analysis Result", width="stretch")
            meta = st.session_state.get("last_meta", {})
            st.info(f"Detections Found: {meta.get('num_boxes', 0)}")
            
            # Show download button
            # ...
        else:
            st.info("Upload an image and run analysis to see results.")

elif page == "History":
    st.title("Patient History")
    
    if st.button("Refresh Data"):
        st.rerun() # or st.experimental_rerun()
        
    try:
        resp = requests.get(f"{API_URL}/history", params={"limit": 20})
        if resp.status_code == 200:
            history = resp.json()
            if history:
                # Convert to dataframe for nicer display
                df = pd.DataFrame(history)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df["detections"] = df["meta"].apply(lambda x: x.get("num_boxes", 0))
                
                # Reorder columns
                display_df = df[["timestamp", "image_key", "detections"]]
                
                st.dataframe(display_df, width='stretch')
                
                # Detail View
                selected_idx = st.selectbox("Select Analysis to View Details", options=range(len(history)), format_func=lambda x: f"{history[x]['timestamp']} - {history[x]['image_key']}")
                
                if selected_idx is not None:
                    item = history[selected_idx]
                    col1, col2 = st.columns(2)
                    with col1:
                        # Fetch image
                        try:
                            img_resp = requests.get(f"{API_URL}/images/{item['image_key']}")
                            if img_resp.status_code == 200:
                                file_bytes = np.asarray(bytearray(img_resp.content), dtype=np.uint8)
                                item_bgr = cv2.imdecode(file_bytes, 1)
                                
                                # Draw
                                item_res_bgr = visualize_prediction_data(
                                    item_bgr,
                                    item["boxes"],
                                    item["labels"],
                                    item["scores"]
                                )
                                st.image(cv2.cvtColor(item_res_bgr, cv2.COLOR_BGR2RGB), caption="Historical Result", width="stretch")
                        except Exception as e:
                            st.error(f"Could not load image: {e}")
                    
                    with col2:
                        st.json(item["meta"])
                        
            else:
                st.warning("No history found.")
        else:
            st.error("Failed to fetch history.")
    except Exception as e:
        st.error(f"Connection error: {e}")
