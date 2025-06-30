import os
import sqlite3
import service
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from User import User
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain import hub
from pydantic import BaseModel, Field
import torch
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.models.whisper")
warnings.filterwarnings("ignore", category=UserWarning, module="langsmith.client")

load_dotenv()

service.init_db()

llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo")

device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device=device
)

SAMPLE_RATE = 16000
DURATION = 10

prompt = hub.pull("hwchase17/openai-functions-agent")


class CreateUserInput(BaseModel):
    name: str = Field(..., description="The user's full name")
    age: int = Field(..., description="The user's age")


@tool
def init_db() -> str:
    """
    Initialize the database.
    """
    service.init_db()
    return "Database initialized."


@tool(args_schema=CreateUserInput)
def create_user(name: str, age: int) -> str:
    """
    Create a new user with the given name and age.
    """
    user = User(name=name, age=age)
    return service.insert_user(user)

@tool
def get_all_users() -> str:
    """
    Retrieve all users from the database.
    """
    users = service.get_all_users()
    if not users:
        return "No users found in the database."
    return "\n".join([f"{user.name}, age {user.age}" for user in users])

agent = create_openai_functions_agent(
    llm=llm,
    tools=[create_user, get_all_users],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[create_user, get_all_users],
    verbose=True
)

print("\n=== Voice Assistant Started ===")
print("Say 'quit' or 'exit' to stop.\n")

while True:
    audio_data = service.record_audio(DURATION, SAMPLE_RATE)
    transcription = whisper_pipeline(
        {
            "sampling_rate": SAMPLE_RATE,
            "raw": audio_data
        },
        generate_kwargs={"language": "ar"}
    )["text"]

    print(f"\nYou said: {transcription}")

    if transcription.strip().lower() in ["quit", "exit"]:
        print("Goodbye!")
        break

    result = agent_executor.invoke({"input": transcription})

    print("\nAssistant:", result["output"])

    if "error" in result:
        print(f"Error: {result['error']}")

    proceed = input("\nDo you want to continue? (yes/no): ").strip().lower()
    if proceed not in ["yes", "y"]:
        print("Session ended.")
        break
