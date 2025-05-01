import streamlit as st
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("API key not found. Please check your .env file.")
else:
    os.environ["OPENAI_API_KEY"] = api_key  # for ChatOpenAI

# Database setup: connect to a SQLite file called alerts.db
engine = create_engine("sqlite:///digital.db")

# OPTIONAL: if you have a CSV, load it into SQLite
# df = pd.read_csv("alerts.csv")
# df.to_sql("alerts", con=engine, if_exists="replace", index=False)

db = SQLDatabase(engine=engine)
llm = ChatOpenAI(model="gpt-4o-mini")
agent_executor = create_sql_agent(
    llm,
    db=db,
    agent_type="openai-tools",
    verbose=True
)

# Data dictionary for context
data_dictionary = """
| Column Name                   | Description                                                                                   |
|-------------------------------|-----------------------------------------------------------------------------------------------|
| alert_id                      | Unique ID for each alert (UUID string)                                                        |
| actionable_entity_id          | Identifier for the related entity (e.g., CTR-1, CTR-2)                                        |
| actionable_entity_nm          | Name of the entity in Arabic (e.g., عقد تشغيل مراكز تحفيظ)                                    |
| actionable_entity_type_nm     | Type of entity (e.g., عقد)                                                                    |
| created_dttm                  | When the alert was created (format DDMMMyy:HH:MM:SS)                                           |
| lstupdt_dttm                  | Last update timestamp (DDMMMyy:HH:MM:SS.ssssss)                                              |
| lstupdt_user_id               | User who last updated the alert (e.g., فهد الحربي)                                            |
| status_dttm                   | When the status last changed (DDMMMyy:HH:MM:SS)                                              |
| alert_status_id               | Status of the alert (e.g., OPEN, CLOSED)                                                      |
| assigned_user_id              | User assigned (e.g., فهد الحربي, ريم العتيبي)                                                  |
| assignment_dttm               | When it was assigned (DDMMMyy:HH:MM:SS)                                                      |
| alert_age                     | Age of the alert in minutes (integer)                                                         |
| Alert_Priority                | Priority (e.g., Low, Medium, High)                                                            |
| lst_refresh_dt                | When the data was last refreshed (DDMMMyy:HH:MM:SS)                                            |
"""

# Streamlit UI setup
st.title("Digital Assistant")
st.write("Ask me anything!")

# Initialize conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# User input box
user_input = st.text_input("You:", key="user_input")

if user_input:
    # Prepend the data dictionary for context
    input_text = f"Refer to the following data dictionary for context:\n\n{data_dictionary}\n\n{user_input}"
    # Invoke the SQL agent
    result = agent_executor.invoke({"input": input_text})["output"]
    # Save into session history
    st.session_state.conversation.append(("You", user_input))
    st.session_state.conversation.append(("Bot", result))
    user_input = ""  # clear after send

# Display the chat history
with st.container():
    for speaker, text in st.session_state.conversation:
        if speaker == "You":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Bot:** {text}")
