import streamlit as st
from elasticsearch import Elasticsearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import streamlit_authenticator as stauth
import uuid
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs.llm_result import LLMResult
from collections import deque
from db import (
    save_conversation,
    save_feedback,
    load_conversations,
)
from prompt_creator import build_prompt, is_query_similar
from prompt_helper import ask_LLM, suggest_categories
import yaml
from yaml.loader import SafeLoader

es_client = Elasticsearch('http://localhost:9200')
index_name = "learning-english"

with open('auth.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

st.title("Language Learning Query System")
st.write("Enter your query below, and the system will search for relevant results in English and Korean, then generate a response.")

def elastic_search(query):
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ['category', 'situation_en^3', 'situation_kr^3', 'question_en', 'answer_en', 'question_kr', 'answer_kr'],
                        "type": "best_fields"
                    }
                }
            }
        }
    }
    response = es_client.search(index=index_name, body=search_query)
    result_docs = [hit['_source'] for hit in response['hits']['hits']]
    return result_docs


# login widget
name, authentication_status, username = authenticator.login(location='main')

# register widget
if not authentication_status:
    if st.button('Register'):
        try:
            if authenticator.register_user('Register user', preauthorization=False):
                st.success('User registered successfully')
        except Exception as e:
            st.error(e)

# load conversations
if authentication_status:
    conversations = load_conversations(username)
    st.sidebar.title("Your Conversations")
    for conversation in conversations:
        st.sidebar.write(conversation["timestamp"])
        st.sidebar.write("----")

if authentication_status:
    st.write(f'### Welcome *{name}*')
    authenticator.logout('Logout', 'main')
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')

# main widget
entry_options = ["please select an option", "pick a category you are interested in", "Any question in your mind?"]
selected_option = st.selectbox("please select an option", entry_options)

if selected_option != "please select an option":
    if selected_option == "pick a category you are interested in":
        suggest_categories()
    else:
        user_query = st.text_input("Any question in your mind?")
        if st.button("Search"):
            if user_query:
                result = is_query_similar(user_query)
                if not result:
                    st.warning("Sorry! I'm here to help with language expressions.\nCould you ask me something related to that?")
                else:
                    if authentication_status:
                        with st.spinner("Searching..."):
                            # ask_LLM
                            answer, token_usage, response_time, search_results = ask_LLM(user_query)
                            st.success("Query Completed!")
                            answer_data = {
                                'answer': answer,
                                'response_time': response_time,
                                'token_usage': token_usage,
                                'search_results': str(search_results)
                            }
                            # st.write(answer)
                            save_conversation(st.session_state.conversation_id, user_query, answer_data, username)
                            col1, col2 = st.columns(2)
                            with col1:
                                pos = st.button('+1')
                            with col2:
                                neg = st.button('-1')
                            if pos:
                                save_feedback(username, st.session_state.conversation_id, 1)
                                st.write("Thanks for your feedback!")
                            if neg:
                                save_feedback(username, st.session_state.conversation_id, -1)
                                st.write("Thanks for your valuable feedback! We will try to improve")
            else:
                st.warning("Please enter a query.")
else:
    st.subheader("Please select an option.")
