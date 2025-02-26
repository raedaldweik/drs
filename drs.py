import streamlit as st
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("API key not found. Please check your .env file.")
else:
    os.environ["OPENAI_API_KEY"] = api_key  # Set API key as environment variable for OpenAI

# Database setup
engine = create_engine("sqlite:///drs.db")
db = SQLDatabase(engine=engine)
llm = ChatOpenAI(model="gpt-4o-mini")
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

# Data dictionary for context
data_dictionary = """
| Column Name                        | Description                                                                                             |
|------------------------------------|---------------------------------------------------------------------------------------------------------|
| Traffic Number                     | Unique identifier for each driver (e.g., 100001, 100002, etc.)                                          |
| Driver Name                        | Name assigned to the driver (e.g., "Driver_42")                                                         |
| Datetime                           | Date and time of the incident in the format YYYY-MM-DD HH:MM:SS                                         |
| Total Fine                         | Fine amount (if the record is a ticket). Blank if the record is an accident                             |
| Location                           | Road name or area of the incident location within Abu Dhabi                                            |
| Latitude                           | Geographic latitude of the incident location (approx. 24.40–24.50)                                      |
| Longitude                          | Geographic longitude of the incident location (approx. 54.30–54.45)                                     |
| Ticket Offence Description         | Description of the traffic violation (if a ticket). Blank if the record is an accident                  |
| Age Range                          | Bucketed age category of the driver (e.g., "18-24", "25-30", etc.)                                      |
| Age                                | Actual integer age of the driver, fitting within the Age Range                                          |
| Gender                             | Driver's gender (e.g., Male, Female)                                                                    |
| Nationality                        | Driver's nationality (e.g., Emirati, Indian, Pakistani, etc.)                                           |
| Driving Experience (years in UAE)  | Number of years the driver has been licensed in the UAE                                                 |
| Record Type                        | Type of record (e.g., "Ticket" or "Accident")                                                           |
| Accident Type                      | Type of accident (e.g., "Rear-end Collision", "Head-on Collision"). Blank if the record is a ticket     |
| Accident Cause                     | Primary cause of the accident (e.g., Speeding, Distracted Driving). Blank if the record is a ticket     |
| Intoxication                       | Indicator (1 for intoxicated, 0 for not intoxicated)                                                    |
| Car Model                          | Make and model of the involved vehicle (e.g., Toyota Camry, BMW 5 Series)                               |
| Car Year                           | Manufacture year of the vehicle (e.g., 2015, 2020)                                                      |
| Car Condition                      | Condition of the vehicle (e.g., Good, Medium, Bad)                                                      |
| Driver Risk Score                  | Calculated driver risk score (range: 300–900) indicating likelihood of future incidents                |
| Road Status                        | Status of the road (e.g., Clear, Busy, Wet)                                                             |
| Weather                            | Weather conditions at the time of incident (e.g., Clear, Rainy, Foggy)                                  |
      |
"""


# Streamlit UI setup
st.title("Digital Assistant")
st.write("Ask me anything!")

# Chatbot conversation state
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# User input
user_input = st.text_input("You:", key="user_input")

if user_input:
    # Add the data dictionary to the input for better context
    input_text = f"Refer to the following data dictionary for context:\n\n{data_dictionary}\n\n{user_input}"
    # Query the RAG model
    result = agent_executor.invoke({"input": input_text})["output"]
    # Append conversation history
    st.session_state.conversation.append(("User", user_input))
    st.session_state.conversation.append(("Bot", result))
    user_input = ""  # Clear input after submission

# Display conversation history in a container with autoscroll enabled
with st.container():
    for speaker, text in st.session_state.conversation:
        if speaker == "User":
            st.write(f"**You:** {text}")
        else:
            st.write(f"**Bot:** {text}")
    # Automatically scrolls to the latest conversation entry
    st_autoscroll = True
