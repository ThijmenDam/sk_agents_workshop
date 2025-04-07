import os
import asyncio
from dotenv import load_dotenv
import PyPDF2
import streamlit as st
from pathlib import Path

# Import Semantic Kernel components
import semantic_kernel as sk
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory, ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions import kernel_function, KernelArguments
from semantic_kernel.connectors.ai import FunctionChoiceBehavior

# Load environment variables from .env file
load_dotenv()

class FileReaderPlugin:
    def __init__(self):
        current_dir = Path(__file__).parent.absolute()
        self.data_dir = current_dir.parent.parent / 'data'
        self.pdf_content = {}
        self._load_pdf_files()

    def _load_pdf_files(self):
        """Load all PDF files from the data directory"""
        data_path = Path(self.data_dir)
        if data_path.exists():
            for pdf_file in data_path.glob('*.pdf'):
                self.pdf_content[pdf_file.name] = None  # Lazy loading

    @kernel_function(description="List available PDF files in the data directory")
    async def list_pdfs(self) -> str:
        """List all available PDF files"""
        files = list(self.pdf_content.keys())
        return "Available PDF files:\n" + "\n".join(files) if files else "No PDF files found"

    @kernel_function(description="Read and search content from PDF files")
    async def read_pdf(self, file_name: str = None) -> str:
        """Read content from a specific PDF file or return error if file not found"""
        if not file_name:
            return await self.list_pdfs()
        
        if file_name not in self.pdf_content:
            return f"File {file_name} not found. {await self.list_pdfs()}"
            
        if self.pdf_content[file_name] is None:  # Lazy load content
            try:
                file_path = os.path.join(self.data_dir, file_name)
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    self.pdf_content[file_name] = text
            except Exception as e:
                return f"Error reading {file_name}: {str(e)}"
        
        return self.pdf_content[file_name]

def create_kernel_with_service():
    kernel = sk.Kernel()
    kernel.add_service(
        AzureChatCompletion(
            service_id="agent",
            deployment_name=os.getenv("AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME")
        )
    )
    return kernel

@st.cache_resource
def initialize_chat():
    """Initialize chat components with caching"""
    kernel = create_kernel_with_service()
    kernel.add_plugin(FileReaderPlugin(), plugin_name="file_reader")
    
    # Configure settings
    settings = kernel.get_prompt_execution_settings_from_service_id("agent")
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
    
    agent = ChatCompletionAgent(
        kernel=kernel,
        name="PDFAssistant",
        instructions="""You are a helpful assistant that can read and analyze PDF documents.
        Use the file_reader plugin to:
        1. List available PDF files using list_pdfs()
        2. Read specific PDF files using read_pdf(file_name)
        3. Answer questions about the content of PDF files
        Be concise and accurate in your responses.""",
        arguments=KernelArguments(settings=settings)
    )
    return agent

async def run_chat(prompt: str):
    """Run the chat interaction"""
    try:
        if st.session_state["agent"] is None:
            st.session_state["agent"] = initialize_chat()
            
        # Initialize chat history 
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = ChatHistory()

        agent = st.session_state["agent"]
        chat_history = st.session_state["chat_history"]
        
        # Add user message to chat history
        chat_history.add_user_message(prompt)

        # Process agent responses
        async for response in agent.invoke(messages=chat_history):
            if response.role == "tool":
                continue
                
            # Get content from response, defaulting to string content
            content = response.content
            if not isinstance(content, str):
                if hasattr(response, 'items') and response.items:
                    content = response.items[0].text if hasattr(response.items[0], 'text') else str(response.items[0])
                elif hasattr(response, 'inner_content') and response.inner_content:
                    content = response.inner_content.choices[0].message.content if response.inner_content.choices else str(response.inner_content)
            
            if not content:
                continue

            # Create message for display and history
            message = {
                "role": response.name.lower() if response.name else "assistant",
                "content": content
            }

            # Display message
            with st.chat_message(message["role"]):
                st.write(message["content"])
            
            # Add to chat history and session state
            chat_history.add_message(ChatMessageContent(
                role=AuthorRole.ASSISTANT,
                content=content,
                name=response.name
            ))
            st.session_state["messages"].append(message)

    except Exception as e:
        st.error(f"Error: {str(e)}")

def main():
    st.title("PDF-Enabled AI Assistant")
    st.caption("Ask questions about PDF documents in the data folder!")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "agent" not in st.session_state:
        st.session_state["agent"] = None

    # Display chat history
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about the PDFs..."):
        # Display user message
        user_message = {"role": "user", "content": prompt}
        st.session_state["messages"].append(user_message)
        with st.chat_message("user"):
            st.write(prompt)

        with st.spinner("Thinking..."):
            # Create new event loop for async operation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(run_chat(prompt))
            finally:
                loop.close()

    # Reset button
    if st.button("New Conversation"):
        st.session_state["messages"] = []
        st.session_state["agent"] = None
        st.rerun()

if __name__ == "__main__":
    main()

