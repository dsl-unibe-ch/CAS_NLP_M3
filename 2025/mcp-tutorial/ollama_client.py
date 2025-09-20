# ollama_client.py
import ollama
import json
import asyncio
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

MODEL = "qwen3:latest" # Using one of the models you have. You need to use a model that accept tools. check ollama models for more
system_msg = {
        "role": "system",
        "content": (
            "You have access to tools. When a tool result is provided, "
            "use it directly to answer the user’s request. "
            "For numbers, state the number clearly. "
            "For text, summarize, explain, or analyze it as needed. "
            "Do not reveal chain-of-thought."
        )
    }


# This function connects to the MCP server to get the list of available tools
async def get_mcp_tools(session: ClientSession) -> list:
    """Fetches tools from the MCP server and formats them for Ollama."""
    print("--- Client: Fetching tools from MCP server... ---")
    tool_list_response = await session.list_tools()
    
    ollama_tools = []
    for tool in tool_list_response.tools:
        ollama_tools.append({
            'type': 'function',
            'function': {
                'name': tool.name,
                'description': tool.description,
                'parameters': tool.inputSchema,
            },
        })
    print(f"--- Client: Loaded {len(ollama_tools)} tools. ---")
    return ollama_tools


async def main():
    """Main loop to run the Ollama client and interact with the MCP server."""
    
    # Define how to start our MCP server as a subprocess
    server_params = StdioServerParameters(
        command="python3",
        args=["mcp_server.py"],
    )

    # Use the stdio_client to manage the server subprocess
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection to the MCP server
            await session.initialize()
            
            # Get the tool definitions from our running server
            tools = await get_mcp_tools(session)

            print("\nOllama MCP Client Initialized. How can I help?")
            print("Model:", MODEL)
            print("Type 'exit' to quit.")
            
            messages = []
            
            while True:
                user_input = input("\n> ")
                if user_input.lower() == 'exit':
                    break
                
                messages.append(system_msg)
                messages.append({'role': 'user', 'content': user_input},)

                # 1. First call to Ollama with tools
                response = ollama.chat(
                    model=MODEL,
                    messages=messages,
                    tools=tools
                )
                messages.append(response['message'])
                
                # 2. Check if the model decided to use a tool
                if response['message'].get('tool_calls'):
                    tool_calls = response['message']['tool_calls']
                    tool_call = tool_calls[0] # Handle one tool call for simplicity
                    tool_name = tool_call['function']['name']
                    tool_args = tool_call['function']['arguments']

                    print(f"--- Client: Model wants to call '{tool_name}' with args: {tool_args} ---")

                    # 3. Execute the tool by calling the MCP server
                    result = await session.call_tool(tool_name, arguments=tool_args)
                    
                    # Extract the text content from the MCP tool result
                    tool_output = ""
                    if result.content and isinstance(result.content[0], types.TextContent):
                        tool_output = result.content[0].text
                    
                    print(f"--- Client: Received tool output: '{tool_output[:100]}...' ---")

                    # 4. Send the tool output back to Ollama
                    messages.append({'role': 'tool', 'content': tool_output})
                    final_response = ollama.chat(model=MODEL, messages=messages)
                    
                    print(f"\nAssistant:\n{final_response['message']['content']}")
                    messages.append(final_response['message'])
                else:
                    # If no tool was called, just print the response
                    print(f"\nAssistant:\n{response['message']['content']}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")