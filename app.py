import streamlit as st
from langchain_core.messages import HumanMessage
from agent import app

st.set_page_config(page_title="AutoStream Assistant", page_icon="🎬", layout="centered")
st.title("🎬 AutoStream AI")
st.markdown("Welcome to AutoStream! Ask me about our video editing plans, pricing, or how to get started.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit_demo_user"

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("E.g., What is the price of the Pro plan?"):

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            last_message = None
            for event in app.stream(
                {"messages": [HumanMessage(content=prompt)]},
                config,
                stream_mode="values"
            ):
                if "messages" in event:
                    last_message = event["messages"][-1]

            ai_response = last_message.content if last_message else "Sorry, something went wrong."
            st.markdown(ai_response)

    st.session_state.messages.append({"role": "assistant", "content": ai_response})