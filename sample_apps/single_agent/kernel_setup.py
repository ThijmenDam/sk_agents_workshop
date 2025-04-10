import os
import streamlit as st

# Import Semantic Kernel components
import semantic_kernel as sk
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from plugin import FileReaderPlugin


def create_kernel_with_service():
    #TODO: Initialize the AzureChatCompletion service with your credentials
    pass 

@st.cache_resource
def initialize_agent() -> ChatCompletionAgent:
    """
    TODO: 
    1) Finish the create_kernel_with_service function to initialize the kernel with the Azure Chat Completion service.
    2) Initialize a kernel and add the file reader plugin.
    3) Configure the settings for the kernel and set the function choice behavior to Auto.
    4) Create an Agent and add it to the kernel with the file reader plugin.
    5) Give the agent a name and instructions for its use.

    AGENT INSTRUCTIONS:
    You are a helpful assistant that can read and analyze PDF documents.
        Use the file_reader plugin to:
        1. List available PDF files using list_pdfs()
        2. Read specific PDF files using read_pdf(file_name)
        3. Answer questions about the content of PDF files
        Be concise and accurate in your responses.
    """
    pass