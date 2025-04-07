import os
from typing import Annotated
from semantic_kernel.functions import kernel_function
from semantic_kernel.contents.utils.author_role import AuthorRole
from azure.ai.projects.models import BingGroundingTool, ToolSet
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential


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
