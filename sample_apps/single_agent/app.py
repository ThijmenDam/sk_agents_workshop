import os
import asyncio
from dotenv import load_dotenv
import streamlit as st
from semantic_kernel.contents import ChatHistory, ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from kernel_setup import initialize_agent


async def run_chat(prompt: str):
    try:
        if st.session_state["agent"] is None:
            print("set agent")
            st.session_state["agent"] = initialize_agent()

        if "chat_history" not in st.session_state:
            print("set history")
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
                    content = response.inner_content.choices[
                        0].message.content if response.inner_content.choices else str(response.inner_content)

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

                chat_message = ChatMessageContent(role=AuthorRole.ASSISTANT, content=content, name=response.name)
                chat_history.add_message(chat_message)

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
    load_dotenv()
    main()
