# semantic-code-mcp (npm wrapper)

Thin `npx` wrapper for the [semantic-code-mcp](https://github.com/huawang1258/semantic-code-mcp) Python MCP server. It launches the PyPI package via `uvx` (or a `pip`-installed console script).

**Prerequisite**: [uv](https://docs.astral.sh/uv/getting-started/installation/) or `pip install semantic-code-mcp`.

## MCP client config

```json
{
  "mcpServers": {
    "semantic-code": {
      "command": "npx",
      "args": ["-y", "semantic-code-mcp"],
      "env": {
        "VOYAGE_API_KEY": "your-voyage-key"
      }
    }
  }
}
```

Full docs: https://github.com/huawang1258/semantic-code-mcp
