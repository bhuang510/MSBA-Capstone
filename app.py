import streamlit as st
import pandas as pd
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine, OperatorConfig
from langchain.agents import Tool, ConversationalChatAgent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks import StreamlitCallbackHandler
import openai
import base64

# Function to display instructions in the sidebar
def display_instructions():
    box_css = """
    <style>
        .instructions-box {
            background-color: #44444;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 20px;
        }
    </style>
    """
    st.sidebar.markdown(box_css, unsafe_allow_html=True)
    st.sidebar.markdown(
        """
    <div class="instructions-box">
        
    ### Background
    The purpose of this application is to demonstrate anonymization of PII in a chatbot. We are using Microsoft Presidio, a Python based module for anonymizing detected PII text entities. 
    """,
        unsafe_allow_html=True,
    )

# Function to convert image to base64 and display logo
def get_image_base64(path):
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

def display_logo():
    image_base64 = get_image_base64("images.png")
    url = 'https://msb.georgetown.edu/msba/'
    html_string = f"""
    <div style="display:flex; justify-content:center;">
        <a href="{url}" target="_blank">
        <img src="data:image/png;base64,{image_base64}" width="150px">
        </a>
    </div>
    """
    st.sidebar.markdown(html_string, unsafe_allow_html=True)



# Initialize Presidio
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Adding a custom recognizer for address
address_recognizer = PatternRecognizer(supported_entity="ADDRESS", patterns=[Pattern("ADDRESS", r"\d{1,5}\s\w+\s\w+\s\w+", 0.5)])
analyzer.registry.add_recognizer(address_recognizer)

# Function to anonymize text
def anonymize_text(text):
    results = analyzer.analyze(text=text, language='en')
    anonymized_text = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators={"DEFAULT": OperatorConfig("replace", {"new_value": "<ANONYMIZED>"})}
    )
    return anonymized_text.text

# Read the CSV file
@st.cache_data
def load_data():
    df = pd.read_csv("PII_Data.csv")
    return df

# Initialize the chatbot tools
def get_anonymized_response(input: str):
    df = load_data()
    results = []
    for index, row in df.iterrows():
        row_str = ' '.join(row.astype(str))
        anonymized_str = anonymize_text(row_str)
        results.append(anonymized_str)
    return '\n'.join(results)

get_anonymized_tool = Tool(
    name='GetAnonymizedResponse',
    func=get_anonymized_response,
    description="Anonymizes and retrieves data from the CSV file."
)

tools = [get_anonymized_tool]

# System message for the agent
system_msg = """
Assistant helps the current user by anonymizing input data to protect personally identifiable information (PII).
"""

# Streamlit app layout
st.set_page_config(page_title="Saxa3 Chatbot")
st.title("Saxa3 Chatbot")

hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Initialize session state
if "langchain_messages" not in st.session_state:
    st.session_state["langchain_messages"] = []

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)

if len(msgs.messages) == 0:
    msgs.clear()

avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        st.write(msg.content)

if prompt := st.chat_input(placeholder="Enter your query:"):
    st.chat_message("user").write(prompt)
    llm = ChatOpenAI(
        model_name="gpt-4-1106-preview",
        temperature=0, streaming=True
    )
    chat_agent = ConversationalChatAgent.from_llm_and_tools(
        llm=llm, tools=tools, verbose=True, system_message=system_msg)
    executor = AgentExecutor.from_agent_and_tools(
        agent=chat_agent,
        tools=tools,
        memory=memory,
        return_intermediate_steps=False,
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=6
    )
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = executor(prompt, callbacks=[st_cb])
        st.write(response["output"])

# Display sidebar content
display_instructions()
display_logo()

