# Azure AI Agents with Semantic Kernel Labs

This repository contains hands-on labs demonstrating the integration of Azure AI Agents Service with Semantic Kernel. The labs are designed to help developers understand and implement AI-powered solutions using Microsoft's latest AI technologies.

## Overview

This project was created during a hackathon to showcase practical applications of Azure AI Agents and Semantic Kernel working together. Through a series of Jupyter notebooks and sample applications, you'll learn how to work with both single agents and multi-agent systems.

The repository includes:
- Interactive Jupyter notebooks that teach the fundamentals of Azure AI Agents and Semantic Kernel
- Step-by-step tutorials progressing from basic single-agent scenarios to complex multi-agent interactions
- Sample applications demonstrating practical implementations of concepts covered in the notebooks
- Complete example scenarios showing real-world applications of AI agents

## Prerequisites

- Azure subscription
- Azure AI Services account
- Python 11.0.0 or later
- Visual Studio Code or Visual Studio 2022
- Basic knowledge of Python and async programming
- Azure CLI installed 

## Repository Structure

- `labs/`: Contains hands-on Jupyter notebooks with interactive tutorials
  - `lab_1.ipynb`: Introduction to Semantic Kernel Agents
    - Basic agent creation and configuration
    - Chat history and agent interactions
    - Function calling and plugins integration
    - Practical exercises with single agents
  - `lab_2.ipynb`: Multi-Agent Systems with Semantic Kernel
    - Transitioning from single to multi-agent systems
    - Agent collaboration using AgentGroupChat
    - Specialized agent roles and team design
    - Agent selection and termination strategies

- `sample_apps/`: Reference implementations demonstrating concepts from labs
  - `single_agent/`: PDF-enabled AI Assistant application
    - Interactive chat interface using Streamlit
    - PDF document analysis and querying
    - Chat history management
    - Plugin integration for document processing
  - `multi_agent/`: Advanced multi-agent implementation
    - Collaborative agent system with specialized roles
    - Complex conversation orchestration
    - Advanced agent selection strategies
  - `solutions/`: Complete reference implementations

- `data/`: Contains sample PDF documents for testing
  - employees.pdf
  - Expanded_Product_Sales_Data.pdf
  - Expenses_Policy.pdf
  - product_data.pdf

## Getting Started

1. Clone this repository
2. run `az login`
2. Configure your Azure credentials
3. Follow the lab instructions in each directory

## Resources and References

### Azure AI Agents
- [Exploring the Semantic Kernel ChatCompletionAgent](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/chat-completion-agent?pivots=programming-language-python)
- [Exploring the Semantic Kernel OpenAIAssistantAgent](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/assistant-agent?pivots=programming-language-python)
- [Exploring the Semantic Kernel AzureAIAgent](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/azure-ai-agent?pivots=programming-language-python)
- [Exploring Agent Collaboration in AgentChat](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/agent-chat?pivots=programming-language-python)

### Semantic Kernel
- [Understanding the kernel](https://learn.microsoft.com/en-us/semantic-kernel/concepts/kernel?pivots=programming-language-python)
- [An Overview of the Agent Architecture](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/agent-architecture?pivots=programming-language-python)
- [Semantic Kernel Components](https://learn.microsoft.com/en-us/semantic-kernel/concepts/semantic-kernel-components?pivots=programming-language-python)
- [Function calling with chat completion](https://learn.microsoft.com/en-us/semantic-kernel/concepts/ai-services/chat-completion/function-calling/?pivots=programming-language-python)

### Additional Resources
- [Develop an Azure AI agent with the Semantic Kernel SDK](https://microsoftlearning.github.io/mslearn-ai-agents/Instructions/04-semantic-kernel.html#create-an-azure-ai-foundry-project)


## License
This project is licensed under the MIT License - see the LICENSE file for details.