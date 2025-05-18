import asyncio
import os
from textwrap import dedent

from agno.agent import Agent
from agno.tools.mcp import MCPTools
from mcp import StdioServerParameters


async def run_agent(message: str) -> None:
    """Run the GitHub agent with the given message."""

    # Initialize the MCP server
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
    )

    # Create a client session to connect to the MCP server
    async with MCPTools(server_params=server_params) as mcp_tools:
        agent = Agent(
            tools=[mcp_tools],
            instructions=dedent("""\
                You are a GitHub assistant. Help users explore repositories and their activity.

                - Use headings to organize your responses
                - Be concise and focus on relevant information\
            """),
            markdown=True,
            show_tool_calls=True,
        )

        # Run the agent
        await agent.aprint_response(message, stream=True)


# Example usage
if __name__ == "__main__":
    while True:
        query = input("> ")
        if query == "q" or query == "quit" or query == "exit":
            os._exit(0)
        asyncio.run(
            run_agent(query)    )