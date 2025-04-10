import os
import streamlit as st

import semantic_kernel as sk
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from plugin import FileReaderPlugin


def create_kernel_with_service(service_id: str):
    kernel = sk.Kernel()
    kernel.add_service(
        AzureChatCompletion(
            service_id=service_id,
            deployment_name=os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT_NAME"),
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
        )
    )
    return kernel


@st.cache_resource
def initialize_agent() -> ChatCompletionAgent:
    service_id = "PDFWizard"

    kernel = create_kernel_with_service(service_id)
    kernel.add_plugin(FileReaderPlugin(), plugin_name='FileReader')

    settings = kernel.get_prompt_execution_settings_from_service_id(service_id=service_id)
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    assistant = ChatCompletionAgent(
        kernel=kernel,
        name="PDFAssistant",
        instructions=
        """
            You are a helpful assistant that can read and analyze PDF documents.
            Use the file_reader plugin to:
            1. List available PDF files using list_pdfs()
            2. Read specific PDF files using read_pdf(file_name)
            3. Answer questions about the content of PDF files
            Be concise and accurate in your responses.
        """,
        arguments=KernelArguments(settings=settings)
    )

    return assistant

