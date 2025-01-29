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

authenticator = stauth.Authenticate("config.yaml")
print(st.session_state)

try:
    authenticator.login()
except LoginError as e:
    st.error(e)

if st.session_state["authentication_status"]:
    st.write('___')
    authenticator.logout()
    print(st.session_state)
    st.write(f'Welcome *{st.session_state["name"]}*')
    st.write('___') 
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')

if st.session_state["authentication_status"]:
    try:
        if authenticator.reset_password(st.session_state["username"]):
            st.success('Password modified successfully')
    except ResetError as e:
        st.error(e)
    except CredentialsError as e:
        st.error(e)
    st.write('_If you use the password reset widget please revert the password to what it was before once you are done._')



# New user registration widget
try:
    (email_of_registered_user,
     username_of_registered_user,
     name_of_registered_user) = authenticator.register_user()
    if email_of_registered_user:
        st.success('User registered successfully')
except RegisterError as e:
    st.error(e)