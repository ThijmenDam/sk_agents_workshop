import streamlit as st
import asyncio
import os
from typing import Annotated
from dotenv import load_dotenv
from sample_apps.multi_agent.kernel_setup import initialize_chat, process_image_and_get_responses



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
