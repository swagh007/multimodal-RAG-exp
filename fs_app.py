
'''
AI Fashion Stylist
'''

import streamlit as st
import requests
import threading
import uvicorn
import time

# ---------- FastAPI Server Setup ---------- #
def run_server():
    import mm_rag  # Import your FastAPI app from mm_rag.py
    uvicorn.run(mm_rag.app, host="0.0.0.0", port=8000)

# Start FastAPI server in background thread
if not hasattr(st, 'server_started'):
    threading.Thread(target=run_server, daemon=True).start()
    st.server_started = True

# ---------- Streamlit UI ---------- #
st.title("AI Fashion Stylist")

with st.form("style_form"):
    query = st.text_input("Enter your style query:")
    gender = st.selectbox("Select Gender", ["Male", "Female", "Other"])
    submitted = st.form_submit_button("Get Styling Ideas")

if submitted:
    if query:
        try:
            # Call local FastAPI endpoint
            response = requests.post(
                "http://localhost:8000/style-query/",
                json={"query": query, "gender": gender}
            )
            
            if response.status_code == 200:
                st.subheader("Styling Suggestions")
                st.markdown(response.json())
            else:
                st.error("Error processing request. Please try again.")
                
        except requests.ConnectionError:
            st.error("Server is starting up... Please wait 5 seconds and try again.")
            time.sleep(5)
    else:
        st.warning("Please enter a style query")
