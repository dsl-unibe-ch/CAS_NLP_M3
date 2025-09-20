import os
import sys
import json
import asyncio
from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

# --- Configuration ---
MODEL = "gpt-4o"  # Or "gpt-3.5-turbo" for a cheaper/faster option

def load_api_key(filepath="api-key") -> str | None:
    """Loads the OpenAI API key from a file."""
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip().startswith("openai="):
                    return line.strip().split('=', 1)[1]
    except FileNotFoundError:
        return None
    return None

# This function connects to the MCP server to get the list of available tools
async def get_mcp_tools(session: ClientSession) -> list:
    """Fetches tools from the MCP server and formats them for OpenAI."""
    print("--- Client: Fetching tools from MCP server... ---")
    tool_list_response = await session.list_tools()
    
    openai_tools = []
    for tool in tool_list_response.tools:
        openai_tools.append({
            'type': 'function',
            'function': {
                'name': tool.name,
                'description': tool.description,
                'parameters': tool.inputSchema,
            },
        })
    print(f"--- Client: Loaded {len(openai_tools)} tools. ---")
    return openai_tools

async def main():
    """Main loop to run the OpenAI client and interact with the MCP server."""
    
    api_key = load_api_key()
    if not api_key:
        print("ERROR: Could not find OpenAI API key.")
        print("Please create a file named 'api-key' with the content 'openai=sk-...'")
        return

    # Initialize the Async OpenAI client
    client = AsyncOpenAI(api_key=api_key)

    # Define how to start our MCP server as a subprocess
    server_params = StdioServerParameters(
        # Use sys.executable to ensure the subprocess uses the same python
        command=sys.executable,
        args=["mcp_server.py"],
    )

    # Use the stdio_client to manage the server subprocess
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection to the MCP server
            await session.initialize()
            
            # Get the tool definitions from our running server
            tools = await get_mcp_tools(session)

            print("\nOpenAI MCP Client Initialized. How can I help?")
            print("Model:", MODEL)
            print("Type 'exit' to quit.")
            
            messages = []
            
            while True:
                user_input = input("\n> ")
                if user_input.lower() == 'exit':
                    break
                
                messages.append({'role': 'user', 'content': user_input})

                # 1. First call to OpenAI with tools
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )
                
                response_message = response.choices[0].message
                messages.append(response_message) # Append the full message object
                
                # 2. Check if the model decided to use a tool
                if response_message.tool_calls:
                    tool_calls = response_message.tool_calls
                    
                    # For this example, we handle one tool call, but you could loop here
                    tool_call = tool_calls[0]
                    tool_name = tool_call.function.name
                    # Arguments are a JSON string, so we need to parse them
                    tool_args = json.loads(tool_call.function.arguments)

                    print(f"--- Client: Model wants to call '{tool_name}' with args: {tool_args} ---")

                    # 3. Execute the tool by calling the MCP server
                    result = await session.call_tool(tool_name, arguments=tool_args)
                    
                    # Extract the text content from the MCP tool result
                    tool_output = ""
                    if result.content and isinstance(result.content[0], types.TextContent):
                        tool_output = result.content[0].text
                    
                    print(f"--- Client: Received tool output: '{tool_output[:100]}...' ---")

                    # 4. Send the tool output back to OpenAI
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": tool_output,
                    })
                    
                    final_response = await client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                    )
                    
                    final_message = final_response.choices[0].message.content
                    print(f"\nAssistant:\n{final_message}")
                    messages.append(final_response.choices[0].message)
                else:
                    # If no tool was called, just print the response
                    assistant_message = response_message.content
                    print(f"\nAssistant:\n{assistant_message}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")