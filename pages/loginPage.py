import streamlit as st
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities import (CredentialsError,
                                               ForgotError,
                                               Hasher,
                                               LoginError,
                                               RegisterError,
                                               ResetError,
                                               UpdateError,
                                               Validator)

# Initialize authentication
authenticator = stauth.Authenticate("config.yaml")

# Application title
st.title("Welcome to the App")

# State management for the current view
if "view" not in st.session_state:
    st.session_state.view = "Login"  # Default to Login view

# Function to switch views (updates session state)
def switch_view(view):
    st.session_state.view = view

# Render the appropriate section based on the current view
if st.session_state.view == "Login":
    st.header("Login")
    try:
        authenticator.login()  # Login widget
    except LoginError as e:
        st.error(e)

    if st.session_state.get("authentication_status"):
        st.success(f'Welcome, *{st.session_state["name"]}*!')
        authenticator.logout()  # Logout button
    elif st.session_state.get("authentication_status") is False:
        st.error('Invalid username or password')
    else:
        st.warning('Please enter your username and password')

elif st.session_state["view"] == "Register":
    st.header("Register New User")
    try:
        email_of_registered_user, username_of_registered_user, name_of_registered_user = authenticator.register_user()
        if email_of_registered_user:
            st.success('User registered successfully')
    except RegisterError as e:
        st.error(e)


if not st.session_state.get("authentication_status"):
    st.write("---")
    if (st.session_state.view == "Register"):
        st.button("Already have an account? Sign in", key="to_login", on_click=lambda: switch_view("Login"))
        switch_view("Login")
    elif (st.session_state.view == "Login"):
        st.button("Dont have an account? Sign up", key="to_register", on_click=lambda: switch_view("Register"))
    


# Optional password reset for authenticated users
if st.session_state.get("authentication_status"):
    st.write("---")
    st.header("Reset Password")
    try:
        if authenticator.reset_password(st.session_state["username"]):
            st.success('Password modified successfully')
    except ResetError as e:
        st.error(e)
    except CredentialsError as e:
        st.error(e)
    st.info('If you reset the password, revert it once done.')
