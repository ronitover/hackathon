import streamlit as st

st.set_page_config(page_title="Hackathon App", page_icon="🚀", layout="wide")

st.title("🚀 Hackathon App")
st.write("Welcome to your Streamlit application.")

name = st.text_input("What's your name?", placeholder="Enter your name")
if name:
    st.success(f"Hello, {name}!")

if st.button("Say hello"):
    st.balloons()
