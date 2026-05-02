from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import os
import json
import re

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Chat Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for ChatGPT-like styling
st.markdown("""
    <style>
        /* Main container styling */
        .main {
            background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
        }
        
        /* Avoid smooth auto-scroll jitter on reruns */
        html, body, [data-testid="stAppViewContainer"] {
            scroll-behavior: auto !important;
        }
        
        /* Chat input styling */
        div[data-testid="stChatInputContainer"] {
            background: linear-gradient(to bottom, #1a1a1a, #0f0f0f);
            border-radius: 12px;
            padding: 10px;
            border: 1px solid #333;
        }

        div[data-testid="stChatInput"] textarea {
            background: #1a1a1a;
            color: white;
            border: 1px solid #333;
            border-radius: 8px;
        }
        
        /* Message styling */
        .user-message {
            background: #10a37f;
            color: white;
            padding: 12px 16px;
            border-radius: 12px;
            margin: 8px 0;
            border-radius: 12px 4px 12px 12px;
        }
        
        .assistant-message {
            background: #444654;
            color: white;
            padding: 12px 16px;
            border-radius: 12px;
            margin: 8px 0;
            border-radius: 4px 12px 12px 12px;
        }
        
        /* Header styling */
        h1 {
            color: #fff;
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 30px;
            background: linear-gradient(135deg, #10a37f 0%, #0f7d5c 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #10a37f 0%, #0f7d5c 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(16, 163, 127, 0.4);
        }
        
        /* Text input styling */
        .stTextInput > div > div > input {
            background: #1a1a1a;
            color: white;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 16px;
        }
        
        .stTextInput > div > div > input::placeholder {
            color: #888;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background: #111;
        }
        
        /* Text color */
        body, .stMarkdown {
            color: #ecedee;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "loading" not in st.session_state:
    st.session_state.loading = False
if "history_loaded" not in st.session_state:
    st.session_state.history_loaded = False

HISTORY_PATH = os.path.join(os.path.dirname(__file__), "chat_history.txt")


def _message_from_role(role, content):
    if role == "human":
        return HumanMessage(content=content)
    return AIMessage(content=content)


def _parse_legacy_history(raw_text):
    pattern = re.compile(r"(HumanMessage|AIMessage)\(content=(['\"])(.*?)\2\)", re.DOTALL)
    parsed_messages = []
    for role, _, content in pattern.findall(raw_text):
        parsed_messages.append(_message_from_role("human" if role == "HumanMessage" else "ai", content))
    return parsed_messages

# Load chat history from file
def load_chat_history():
    if st.session_state.history_loaded:
        return
    if not os.path.exists(HISTORY_PATH):
        st.session_state.history_loaded = True
        return

    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    parsed_messages = []
    for line in content.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "role" in obj and "content" in obj:
                parsed_messages.append(_message_from_role(obj["role"], obj["content"]))
        except json.JSONDecodeError:
            parsed_messages = []
            break

    if not parsed_messages and content.strip():
        parsed_messages = _parse_legacy_history(content)

    st.session_state.chat_history.extend(parsed_messages)
    st.session_state.history_loaded = True

# Save message to file
def save_message(role, content):
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"role": role, "content": content}, ensure_ascii=False) + "\n")

# Initialize chat model
chat_model = ChatOllama(
    model="gemma4:31b-cloud",
    temperature=0.7,
)

chat_template = ChatPromptTemplate([
    ('system', 'you are a helpful customer support assistant'),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human', '{query}')
])

# Load initial history
load_chat_history()

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    model_temp = st.slider("Temperature", 0.0, 2.0, 0.7, step=0.1)
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        with open(HISTORY_PATH, 'w', encoding="utf-8") as f:
            f.write("")
        st.rerun()
    
    st.markdown("---")
    st.markdown("**About**\n\nCustomer Support Assistant powered by LLM")

# Main header
st.markdown("<h1>💬 Chat Assistant</h1>", unsafe_allow_html=True)

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)
prompt = st.chat_input("Ask me anything...")

if prompt and prompt.strip():
    user_input = prompt.strip()

    st.session_state.chat_history.append(HumanMessage(content=user_input))
    save_message("human", user_input)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            chain = chat_template | chat_model
            response = chain.invoke({
                "chat_history": st.session_state.chat_history,
                "query": user_input
            })
            assistant_response = response.content
            st.markdown(assistant_response)

    st.session_state.chat_history.append(AIMessage(content=assistant_response))
    save_message("ai", assistant_response)

st.markdown("""
    <style>
        .stChatMessage {
            padding: 16px;
            margin: 8px 0;
        }
    </style>
""", unsafe_allow_html=True)
