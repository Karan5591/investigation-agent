import streamlit as st
import requests

API_URL = "http://localhost:8001/investigate"

st.title("Investigation Agent")

question = st.text_area("Ask a question", "Did the outage qualify for a service credit review?")
top_k = st.slider("Top K", 1, 10, 5)

if st.button("Run investigation"):
    payload = {"question": question, "top_k": top_k}
    resp = requests.post(API_URL, json=payload, timeout=120)
    data = resp.json()

    st.subheader("Result")
    st.write(data.get("status"))
    st.write(data.get("answer"))
    st.write(f"Confidence: {data.get('confidence')}")

    if data.get("warnings"):
        st.warning("\n".join(data["warnings"]))

    for item in data.get("evidence", []):
        st.markdown(f"**{item['filename']}** (page {item.get('page')})")
        st.write(item["snippet"])