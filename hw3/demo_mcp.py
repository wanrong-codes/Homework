from agent.mcp_client import get_default_mcp_server_params, load_mcp_tools


def main():
    server_params = get_default_mcp_server_params()
    print(f"连接 MCP server: {server_params.command} {' '.join(server_params.args)}")

    tools = load_mcp_tools(server_params)
    print(f"\n通过 MCP list_tools() 发现 {len(tools)} 个工具：")
    for t in tools:
        print(f"  - {t.name}: {t.description}")

    add_tool = next((t for t in tools if t.name.endswith("add_note")), None)
    list_tool = next((t for t in tools if t.name.endswith("list_notes")), None)

    if add_tool and list_tool:
        print("\n--- 通过 MCP call_tool() 调用 add_note ---")
        print(add_tool.run(text="完成 hw3 的 MCP 集成"))

        print("\n--- 通过 MCP call_tool() 调用 list_notes ---")
        print(list_tool.run())


if __name__ == "__main__":
    main()
