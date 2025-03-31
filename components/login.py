import streamlit as st
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities import (LoginError, RegisterError, ResetError, CredentialsError)


class Login():
    def __init__(self, permission:bool=True) -> None:

        self.authenticator = stauth.Authenticate("config.yaml")

        if "view" not in st.session_state:
            st.session_state.view = "Login"  # Default to Login view

        # Render the appropriate section based on the current view
        if st.session_state.view == "Login":
            try:
                self.authenticator.login()  # Login widget
            except LoginError as e:
                st.error(e)

            if st.session_state.get("authentication_status"):
                st.success(f'Welcome, *{st.session_state["name"]}*!')
                if permission: self.authenticator.logout()  # Logout button
            elif st.session_state.get("authentication_status") is False:
                st.error('Invalid username or password')
            else:
                st.warning('Please enter your username and password')

        elif st.session_state.view == "Register":
            try:
                email_of_registered_user, username_of_registered_user, name_of_registered_user = self.authenticator.register_user()
                if email_of_registered_user:
                    st.success('User registered successfully')
            except RegisterError as e:
                st.error(e)


        if not st.session_state.get("authentication_status"):
            st.write("---")
            if (st.session_state.view == "Register"):
                st.button("Already have an account? Sign in", key="to_login", on_click=lambda: self.__switch_view("Login"))
            elif (st.session_state.view == "Login"):
                st.button("Dont have an account? Sign up", key="to_register", on_click=lambda: self.__switch_view("Register"))
        # else:
        #     st.switch_page("pages/importPage.py")

    def resetPassword(self) -> None:
        # Optional password reset for authenticated users
        if "reset_pswd" not in st.session_state:
            st.session_state.reset_pswd = False

        if st.session_state.get("authentication_status"):
            if st.session_state.reset_pswd:
                st.write("---")
                st.header("Reset Password")
                try:
                    if self.authenticator.reset_password(st.session_state["username"]):
                        st.success('Password modified successfully')
                except ResetError as e:
                    st.error(e)
                except CredentialsError as e:
                    st.error(e)
                st.info('If you reset the password, revert it once done.')

                st.button("Hide", key="hide", on_click=lambda: self.__reset_pswd(False))
            
            else:
                st.button("Change Password", key="to_reset_pswd", on_click=lambda: self.__reset_pswd(True))


    # Function to switch views (updates session state)
    def __switch_view(self, view:str) -> None:
        st.session_state.view = view

    def __reset_pswd(self, vl:bool) -> None:
        st.session_state.reset_pswd = vl