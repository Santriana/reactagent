import streamlit as st
from modules.agent.vectordb_react import CustomOutputParser, queryreactagent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate


def get_response(user_query, chat_history):
    agent_instance = queryreactagent()
    response = agent_instance.ask(user_query)
    
    if isinstance(response, dict):
        response_content = response.get("output", "")
    else:
        response_content = str(response)
    
    return response_content

def main():
    st.set_page_config(page_title="Siloam Hospital Virtual Assistant", page_icon="🤖")
    st.title("Siloam Hospital Virtual Assistant")
    st.subheader('''
                Ask anything about Siloam Hospital Clinical Pathway\n
                i.e\n
                What is the comparison of total admissions at Siloam Hospital Lippo Village from 2020 - 2023?''')

    # Session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am Siloam Hospital Virtual Assistant. How can I help you?"),
        ]

    # Conversation display
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)

    # User input
    user_query = st.chat_input("Type your message here...")
    if user_query:
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            ai_response = get_response(user_query, st.session_state.chat_history)
            st.markdown(ai_response)

        st.session_state.chat_history.append(AIMessage(content=ai_response))

if __name__ == "__main__":
    main()
