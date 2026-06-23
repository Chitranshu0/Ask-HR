'''
This page is set-up for the agentic feature Chatbot feature of the chat, where our main component the /chatbot is going to setup. 
'''


from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

from langgraph.graph import StateGraph, START, END

from langchain_groq import ChatGroq

from dotenv import load_dotenv
import asyncio 


load_dotenv()


###############
# Chatbot State
###############
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


###############
# LLM Defined 
###############
llm = ChatGroq(model='llama-3.1-8b-instant', temperature=.1, max_tokens=300)


def chat_node(state: ChatState)-> ChatState:
    '''
    The message is send to the llm. 
    And the reponse is saved to the State.
    '''
    messages = state['messages']
    response = llm.invoke(messages)
    return {
        'messages': [response]
    }



async def agentic_chatbot():
    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)

    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)

    chatbot = graph.compile()
    return chatbot







