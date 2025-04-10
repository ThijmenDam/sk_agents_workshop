import io
import streamlit as st
import asyncio
import os
from typing import Annotated
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent, AgentGroupChat
from semantic_kernel.agents.strategies import KernelFunctionSelectionStrategy, KernelFunctionTerminationStrategy, DefaultTerminationStrategy
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.functions import kernel_function, KernelArguments, KernelFunctionFromPrompt
from semantic_kernel.contents import ChatHistory, ChatMessageContent, ImageContent, TextContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from azure.ai.projects.models import BingGroundingTool, ToolSet
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
import aiohttp


load_dotenv()

# Update the agent instructions
SEARCHER_NAME = "ProductSearchAgent"
ANALYZER_NAME = "ImageAnalyzerAgent"

# Agent instructions as module-level constants
SEARCHER_INSTRUCTIONS = """
You are a product search specialist. Your role is to find purchasable items based on detailed image analysis.

Your tasks:
1. Use the provided image analysis from the ImageAnalyzerAgent to understand the product details.
2. Extract key search terms and attributes from the analysis, focusing on:
   - Primary object type and category
   - Material, color, style, and approximate size
   - Distinctive features and design elements
   - Quality indicators and brand style (if apparent)

3. Formulate effective search queries using these details to find similar products across online retailers.

4. Return the top 5 most relevant product matches, prioritizing:
   - Visual similarity to the analyzed image
   - Matching features and specifications
   - Competitive pricing and availability
   - Product quality and ratings
   - Brand reputation (when relevant)

5. Present results in a clear, organized format:
   - Product name with direct shopping link
   - Price and retailer information
   - Key features that match the analyzed image
   - Any notable variations available
   - Match confidence level (exact match vs. similar alternative)

Remember: Your goal is to help users find products that closely match what they see in their image. Use the detailed image analysis to ensure accurate matches.
"""

ANALYZER_INSTRUCTIONS = """
You are an image analysis expert. When analyzing images:

1. Use the extract_image_from_message function to analyze the image, which will be provided in the items of the message.
   Decode the Image Content to get the image bytes and analyze it using Azure Computer Vision.

2. Based on the analysis, provide a clear, detailed description that includes:
   - The main object(s) and their distinguishing features
   - Notable characteristics such as shape, color, texture, material, and size
   - Style and design elements (e.g., vintage, modern, minimalist)
   - Functional or decorative details relevant to the object

3. Use objective, neutral, and factual language. Avoid opinions or assumptions.
4. Write descriptions optimized for product searchability (e.g., including common keywords or terms a user might search for).

Example format:
"A round, matte-black ceramic bowl with a minimalist design, featuring a smooth finish and slightly flared rim."

Be thorough yet concise. Focus on what would be most useful to someone browsing or searching for this item online.
"""


class BingSearchPlugin:
    def __init__(self):
        # Get the connection from the project client first
        self.project_client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(exclude_environment_credential=True, exclude_managed_identity_credential=True),
            conn_str=os.getenv("AZURE_AI_AGENT_PROJECT_CONNECTION_STRING")
        )
        bing_connection = self.project_client.connections.get(
            connection_name=os.getenv("BING_CONNECTION_NAME"),
            include_credentials=True
        )
        self._tool = BingGroundingTool(connection_id=bing_connection.id)
    
    @kernel_function(description="Search Bing for real-time web results")
    async def search(self, query: Annotated[str, "The search query"]) -> Annotated[str, "Search results"]:
        toolset = ToolSet()
        toolset.add(self._tool)
        
        # Create agent with search capabilities
        agent = self.project_client.agents.create_agent(
            model=os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT_NAME"),
            instructions="Search for products based on the image caption and provide links to buy them.",
            toolset=toolset
        )
        
        try:
            # Create thread and message
            thread = self.project_client.agents.create_thread()
            self.project_client.agents.create_message(
                thread_id=thread.id,
                role="user",
                content=f"buy {query} price availability shop"
            )
            
            # Process the search
            run = self.project_client.agents.create_and_process_run(
                thread_id=thread.id,
                agent_id=agent.id
            )
            
            if run.status == "completed":
                messages = self.project_client.agents.list_messages(thread_id=thread.id)
                response = messages.get_last_text_message_by_role("assistant")
                return response.text.value if response else "No results found"
            else:
                return f"Search failed: {run.last_error}"
                
        finally:
            # Cleanup
            self.project_client.agents.delete_agent(agent.id)



class ImageAnalysisPlugin:
    """A plugin that analyzes images using Azure Computer Vision."""
    def __init__(self):
        self.client = ImageAnalysisClient(
            endpoint=os.environ["COMPUTER_VISION_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["COMPUTER_VISION_KEY"]),
        )   
                    
    async def download_image_bytes(self, url: str) -> bytes:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    raise Exception(f"Failed to download image. Status code: {response.status}")

    async def extract_image_from_message(self, message: Annotated[ImageContent, "Latest message from the user with the image"]) \
            -> Annotated[str, "Image description"]:

        try:    
            uri = str(message.uri)
            image_bytes = await self.download_image_bytes(uri)

            # Analyze image
            result = self.client.analyze(
                image_data=image_bytes,
                visual_features=[
                    VisualFeatures.CAPTION,
                    VisualFeatures.TAGS,
                    VisualFeatures.OBJECTS,
                    VisualFeatures.DENSE_CAPTIONS
                ],
                gender_neutral_caption=True,
            )

            # Build response
            description_parts = []

            if result.caption:
                description_parts.append(result.caption.text)

            if result.tags:
                relevant_tags = [tag.name for tag in result.tags[:5]]
                description_parts.append(f"Key features: {', '.join(relevant_tags)}")

            if result.dense_captions:
                detailed_captions = [cap.text for cap in result.dense_captions[:3]]
                description_parts.append("Additional details: " + "; ".join(detailed_captions))

            return " ".join(description_parts) if description_parts else "No description could be generated for this image."

        except Exception as e:
            print(f"Error analyzing image: {str(e)}")

            
@st.cache_resource
def create_kernel():
    kernel = Kernel()
    
    # Add the service with the configured settings
    kernel.add_service(
        AzureChatCompletion(
            service_id="agent",
            deployment_name=os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT_NAME"),
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        )
    )
    
    # Add image plugin with explicit function registration
    image_plugin = ImageAnalysisPlugin()
    kernel.add_plugin(
        plugin=image_plugin,
        plugin_name="image"
    )
    
    # Add search plugin with function calling enabled
    kernel.add_plugin(BingSearchPlugin(), plugin_name="bing")
    
    return kernel

async def initialize_chat():
    kernel = create_kernel()
    
    # Create search agent with proper function calling behavior
    search_agent = ChatCompletionAgent(
        kernel=kernel,
        name=SEARCHER_NAME,
        instructions=SEARCHER_INSTRUCTIONS,
        function_choice_behavior=FunctionChoiceBehavior.Auto()
    )

    # Create image analyzer agent with function calling behavior
    image_agent = ChatCompletionAgent(
        kernel=kernel,
        name=ANALYZER_NAME,
        instructions=ANALYZER_INSTRUCTIONS,
        function_choice_behavior=FunctionChoiceBehavior.Auto()
    )

    selection_function = KernelFunctionFromPrompt(
        function_name="selection",
        prompt=f"""
        Determine which participant takes the next turn in a conversation based on the most recent participant.
        State only the name of the participant to take the next turn.
        No participant should take more than one turn in a row.

        Choose only from these participants:
        - {ANALYZER_NAME}
        - {SEARCHER_NAME}

        Always follow these rules when selecting the next participant:
        - After user input, it is {ANALYZER_NAME}'s turn to extract the image from the user's message and analyze it.
        - After {ANALYZER_NAME} describes the image, it is {SEARCHER_NAME}'s turn to search for products.
        - After {SEARCHER_NAME} provides product options, end the conversation.
        
        History:
        {{$history}}
        """
    )
    print("sddds")
    # Create the group chat with updated selection strategy
    chat = AgentGroupChat(
        agents=[search_agent, image_agent],
        selection_strategy=KernelFunctionSelectionStrategy(
            function=selection_function,
            kernel=kernel,
            result_parser=lambda result: (
                # Handle both list and string results
                str(result.value[0]).strip() if isinstance(result.value, list) and result.value 
                else result.value.strip() if isinstance(result.value, str) 
                else None
            ),
            agent_variable_name="agents",
            history_variable_name="history"
        ),
        termination_strategy=DefaultTerminationStrategy(maximum_iterations=4)
    )
    
    # Initialize empty chat history
    chat.history = ChatHistory()
    
    return chat


async def process_image_and_get_responses(chat, image_bytes):
    try:
        print(image_bytes)
        analyze_message = ChatMessageContent(
            role=AuthorRole.USER,
            content="Please analyze this image",
            items=[ImageContent(uri=image_bytes)],
            metadata={"image_data": image_bytes}
        )
        
        # print(f"Message content type: {type(analyze_message.items[0].data)}")

        # print(f"MESSAGE TYPE: {type(analyze_message)}")
        # print(f"MESSAGE INNER CONTENT: {type(analyze_message.metadata['image_data'])}")

        # Display initial message in Streamlit 
        with st.chat_message("user"):
            st.write("Find listings of the object in the image:")
            st.image(image_bytes)

        # Add message to chat history
        await chat.add_chat_message(analyze_message)  

        # async for x in chat.get_chat_messages():
        #     print(f"Chat message: {x.role} - {x.content} - {x.items[0].data}")

        async for response in chat.invoke():
            if response.role != "tool":
                # Handle string or dict content types
                content = response.content
                if not isinstance(content, str):
                    content = content.get("text", str(content))

                response_dict = {
                    "role": response.name.lower() if response.name else "assistant",
                    "content": content
                }
                
                with st.chat_message(response_dict["role"]):
                    st.write(response_dict["content"])
                st.session_state["messages"].append(response_dict)
                
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        print(f"Detailed error in process_image_and_get_responses: {str(e)}")


def main():
    st.set_page_config(page_title="Image Analysis & Shopping Assistant")

    st.title("Image Analysis & Shopping Assistant")
    st.caption("Enter an image URL to analyze")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "chat" not in st.session_state:
        st.session_state["chat"] = None

    # Display chat history
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "image_path" in message and message["image_path"]:
                st.image(message["image_path"])

    # Simple URL input field
    image_url = st.text_input("Enter image URL:")
    
    if image_url:
        try:
            # Display the image from URL
            st.image(image_url, caption="Image from URL", use_container_width=True)
            
            # Add an analyze button
            if st.button("Analyze and Find Products"):
                with st.spinner("Processing image from URL..."):
                    # Initialize chat if not already initialized
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        # Initialize chat if needed
                        if st.session_state["chat"] is None:
                            st.session_state["chat"] = loop.run_until_complete(initialize_chat())
                        
                        # Process image analysis and chat responses using URL
                        loop.run_until_complete(process_image_and_get_responses(
                            st.session_state["chat"],
                            image_url
                        ))
                    finally:
                        loop.close()
        except Exception as e:
            st.error(f"Error loading image from URL: {str(e)}")

    # Reset button
    if st.button("Start Over"):
        st.session_state.clear()
        st.rerun()

if __name__ == "__main__":
    load_dotenv()
    main()
