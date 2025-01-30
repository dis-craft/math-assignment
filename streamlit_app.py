# pip install -U streamlit
# streamlit run streamlit_app.py

import streamlit as st

st.title("This is a title")
st.header("This is a header")
st.subheader("This is a subheader")
st.text("Text data")
# st.tk.Checkbutton(root, text="HELLO")
st.write("This is a default font")
st.markdown("This a markdown!")

st.markdown("# Markdown heading")
st.markdown("### Markdown heading")
st.markdown("**Markdown heading in bold**")
st.markdown("""`code Hello world 
hello again` """)
st.markdown('''
1. First item
2. Second item
3. Third item
            ''')