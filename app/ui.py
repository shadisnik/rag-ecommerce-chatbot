import streamlit as st
import requests

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🛍️ E-commerce RAG Chatbot")
st.write("Ask about products below.")


def render_product_cards(products):
    if not products:
        return

    cols_per_row = 3

    for i in range(0, len(products), cols_per_row):
        cols = st.columns(cols_per_row)

        for col, p in zip(cols, products[i:i + cols_per_row]):
            with col:
                with st.container(border=True):
                    if p.get("image_path"):
                        st.image(p["image_path"], use_container_width=True)

                    st.markdown(f"#### {p.get('product_name', 'Unknown Product')}")
                    st.write(f"**Category:** {p.get('category', 'N/A')}")
                    st.write(f"**Color:** {p.get('color', 'N/A')}")
                    st.write(f"**Usage:** {p.get('usage', 'N/A')}")

                    if p.get("brand"):
                        st.write(f"**Brand:** {p.get('brand', 'N/A')}")

                    if p.get("price"):
                        st.write(f"**Price:** {p.get('price', 'N/A')}")

                    if p.get("link"):
                        st.link_button("🛒 Buy now", p["link"])


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["content"]:
            st.markdown(msg["content"])

        if msg["role"] == "assistant" and "retrieved_products" in msg:
            render_product_cards(msg["retrieved_products"])

user_input = st.chat_input("Ask me about products...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        response = requests.post(
            "http://127.0.0.1:8000/chat",
            json={"query": user_input},
            timeout=120
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("found"):
            bot_reply = data.get("message", "No answer found.")
        else:
            bot_reply = ""

        assistant_msg = {
            "role": "assistant",
            "content": bot_reply,
            "retrieved_products": data.get("retrieved_products", [])
        }

    except Exception as e:
        assistant_msg = {
            "role": "assistant",
            "content": f"Error connecting to API: {e}",
            "retrieved_products": []
        }

    st.session_state.messages.append(assistant_msg)

    with st.chat_message("assistant"):
        if assistant_msg["content"]:
            st.markdown(assistant_msg["content"])
        render_product_cards(assistant_msg["retrieved_products"])