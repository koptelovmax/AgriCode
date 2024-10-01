import streamlit as st
import base64

def set_header():
    LOGO_IMAGE = "agriloop-logo.png"

    st.markdown(
        """
        <style>
        .container {
            display: flex;
        }
        .logo-text {
            font-weight:700 !important;
            font-size:50px !important;
            color: #f9a01b !important;
            padding-left: 10px !important;
        }
        .logo-img {
            float:right;
            width: 28%;
            height: 28%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        f"""
        <div class="container">
            <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
            <p class="logo-text">AgriCode</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
set_header()

st.write("Select classification method:")

if st.button("Paragraphs"):
    st.switch_page("pages/Paragraphs.py")
if st.button("Sentences"):
    st.switch_page("pages/Sentences.py")
