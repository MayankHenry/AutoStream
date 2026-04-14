import streamlit as st
from langchain_core.messages import HumanMessage
from agent import app

st.set_page_config(page_title="AutoStream Assistant", page_icon="🎬", layout="centered")

st.title("🎬 AutoStream AI Assistant")
st.markdown("Ask me about pricing, features, or get started with a plan!")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user_session_1"

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("E.g. What's included in the Pro plan?"):

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

            if last_message is None:
                ai_response = "Sorry, something went wrong. Please try again."
            elif hasattr(last_message, "content"):
                ai_response = last_message.content
            else:
                ai_response = str(last_message)

            st.markdown(ai_response)

    st.session_state.messages.append({"role": "assistant", "content": ai_response})