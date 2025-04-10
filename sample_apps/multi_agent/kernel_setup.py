import streamlit as st
import os
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent, AgentGroupChat
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.functions import KernelFunctionFromPrompt
from semantic_kernel.contents import ChatHistory, ChatMessageContent, ImageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from plugins.bing_search_plugin import BingSearchPlugin
from plugins.image_plugin import ImageAnalysisPlugin
from config.analysis_agent_config import  ANALYZER_NAME, ANALYZER_INSTRUCTIONS
from config.search_agent_config import SEARCHER_NAME, SEARCHER_INSTRUCTIONS


@st.cache_resource
def create_kernel():
    kernel = Kernel()
    
    # TODO: Add the service with the configured settings
    # 1. Add a chat completion service with the configured Azure settings

    # TODO: Add image plugin with explicit function registration

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
        - After user input, it is {ANALYZER_NAME}'s turn to analyze the image.
        - After {ANALYZER_NAME} describes the image, it is {SEARCHER_NAME}'s turn to search for products.
        - After {SEARCHER_NAME} provides product options, end the conversation.
        
        History:
        {{$history}}
        """
    )

    #TODO: Create the group chat with updated selection strategy
    """
    1. Add the two agents to the chat group
    2. set the following result parser:
    result_parser=lambda result: (
        # Handle both list and string results
        str(result.value[0]).strip() if isinstance(result.value, list) and result.value 
        else result.value.strip() if isinstance(result.value, str) 
        else None
    ),
    3. Add the selection strategy to the group chat
    4. Add the default termination strategy to the group chat
    """
    chat = AgentGroupChat()
    
    #TODO: Initialize empty chat history
    
    
    return chat

async def process_image_and_get_responses(chat, image_bytes):
    try:
        # Create the image content
        image_content = ImageContent(
            data=image_bytes,
        )
        print(f"Image content type: {type(image_content.data)} {image_content.data}")

        # Create chat message with proper structure
        analyze_message = ChatMessageContent(
            role=AuthorRole.USER,
            content="Please analyze this image",
            metadata={"image_data":image_content.data} 
        )

        print(f"MESSAGE TYPE: {type(analyze_message)}")
        print(f"MESSAGE INNER CONTENT: {type(analyze_message.metadata['image_data'])}")

        # Display initial message in Streamlit 
        with st.chat_message("user"):
            st.write("Analyzing your image...")
            st.image(image_bytes)

        # Add message to chat history
        chat.history.add_message(analyze_message)  
        print(f"Chat history after adding message: {chat.history}")


        # Invoke chat with proper image content
        async for response in chat.invoke():
            if response.role != "tool":
                # Handle string or dict content types
                content = response.content
                print(f"Response type: {type(response)}")
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
        # Don't re-raise to allow graceful error handling
