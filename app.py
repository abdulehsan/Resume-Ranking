import streamlit as st

Models_page = st.Page("Models.py", title="Rank using different Algos", icon=":material/add_circle:")
AI_rank = st.Page("AI_rank.py", title="Rank with AI", icon=":material/delete:")

pg = st.navigation([Models_page, AI_rank])
st.set_page_config(page_title="Home", page_icon=":material/edit:")
pg.run()