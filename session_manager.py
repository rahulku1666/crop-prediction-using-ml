import streamlit as st

class SessionManager:
    @staticmethod
    def init_session():
        if 'logged_in' not in st.session_state:
            st.session_state.logged_in = False
        if 'username' not in st.session_state:
            st.session_state.username = None

    @staticmethod
    def login_user(username):
        st.session_state.logged_in = True
        st.session_state.username = username

    @staticmethod
    def logout_user():
        st.session_state.logged_in = False
        st.session_state.username = None

    @staticmethod
    def is_logged_in():
        return st.session_state.logged_in