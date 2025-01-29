import streamlit as st
import streamlit_authenticator as stauth

def getUserContext():
    stauth.Authenticate("config.yaml").login()