import streamlit as st
from st_pages import add_page_title, get_nav_from_toml

st.set_page_config(layout="wide")

# sections = st.sidebar.toggle("Sections", value=True, key="use_sections")

nav = get_nav_from_toml(".streamlit/pages_sections.toml")

pg = st.navigation(nav)

add_page_title(pg)

pg.run()

st.markdown("""
    <script>
        if (performance.navigation.type === 1) {
            window.location.replace("/");
        }
    </script>
""", unsafe_allow_html=True)
if "page_loaded" not in st.session_state:
    st.session_state.page_loaded = True
    st.switch_page("pages/homePage.py")  # Redirect to home page on refresh