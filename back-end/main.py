from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
from langchain_openai import ChatOpenAI
from IPython.display import Image, display
from graph import getBuilder
from dotenv import load_dotenv  # Import load_dotenv
from utilities import _print_event
import json

# Load environment variables from .env file
load_dotenv()
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# Initialize FastAPI and LangChain (with OpenAI model)
app = FastAPI()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OpenAI API key is not set")

# Initialize LangChain OpenAI language model
# llm = OpenAI(api_key=openai_key)
llm = ChatOpenAI(model="gpt-4o-mini")

# Define request and response data models
class QueryRequest(BaseModel):
    query: str
    thread_id: int

class QueryResponse(BaseModel):
    response: str

graph = getBuilder()

@app.get("/")
async def read_root():
    logging.debug("Root endpoint was called")
    return {"message": "Welcome to the FastAPI + LangChain API"}

@app.get("/graph")
async def generateGraph():
    try:
        logging.info("Root endpoint was called")
        graph = getBuilder()
        display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
    except Exception:
        # This requires some extra dependencies and is optional
        logging.info("Exception Root endpoint was called", Exception)
        pass

# Endpoint for language model queries
@app.post("/query")
async def query_model(request: QueryRequest):
    try:
        # Process the query using LangChain (OpenAI model)
        # response = llm(request.query)
        logging.info("query", request.query)
        logging.info("thread_id", request.thread_id)

        config = {
            "configurable": {
                # The passenger_id is used in our flight tools to
                # fetch the user's flight information
                "passenger_id": "3442 587242",
                # Checkpoints are accessed by thread_id
                "thread_id": request.thread_id,
            }
        }

        snapshot = graph.get_state(config)
        if snapshot.next:
            logging.info(" User Ok ")
            result = graph.invoke(
                None,
                config,
            )
            
            return { "msg": result.get('messages')[-1].content }

        _printed = set()
        events = graph.stream(
            {"messages": ("user", request.query)}, config, stream_mode="values"
        )
        logging.info("logging-events-------", events)
        for event in events:
            msg=_print_event(event, _printed)
        snapshot = graph.get_state(config)
        if snapshot.next:
            logging.info(" Need User Confirm ")
            return { "msg": snapshot.values["messages"][-1] }
        # snapshot = graph.get_state(config)
        # while snapshot.next:
        #     logging("comfiromation: ", snapshot.values["messages"][-1])

        # return QueryResponse(response=response)
        # json_string = json.dumps(list(events.output()))
       
        return { "msg": msg }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
