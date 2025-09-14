import asyncio
import logging
import platform
import time
from typing import Annotated

import pytest
from pydantic import Field, TypeAdapter
import anthropic
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from fastmcp import FastMCP
import agtk
from agtk.types import ToolParamSchema

logger = logging.getLogger(__name__)


# Test toolkit with various tool patterns
class SystemInfoToolkit(agtk.Toolkit):
    """Test toolkit with system information tools."""

    @agtk.tool_def()
    def get_system_architecture(self):
        """Gets system architecture with async wrapper pattern."""
        import platform

        async def async_wrapper():
            return platform.machine()

        result = asyncio.run(async_wrapper())
        return result

    @agtk.tool_def()
    def get_timezone(self):
        """Gets timezone information."""
        return time.tzname[0]

    @agtk.tool_def()
    def get_platform_info(
        self,
        include_version: Annotated[
            bool, Field(description="Include version info")
        ] = False,
    ):
        """Gets platform information with optional parameter."""
        info = {"system": platform.system()}
        if include_version:
            info["version"] = platform.version()
        return info


class AsyncToolkit(agtk.Toolkit):
    """Test toolkit with async tools."""

    @agtk.tool_def()
    async def async_operation(self, delay: float = 0.001):
        """Performs async operation with delay."""
        await asyncio.sleep(delay)
        return f"Completed after {delay}s"

    @agtk.tool_def()
    def sync_operation(self, value: str):
        """Performs sync operation."""
        return f"Processed: {value}"


# Helper functions
def create_toolkit_tool(toolkit_cls):
    """Helper to create toolkit instance."""
    return toolkit_cls()


def validate_toolkit_schema(toolkit, schema_type, expected_tool_count=None):
    """Helper to validate toolkit schema with TypeAdapter."""
    params = toolkit.as_param(schema_type)

    if expected_tool_count is not None:
        assert len(params) == expected_tool_count

    # Validate each tool param with TypeAdapter
    if schema_type == ToolParamSchema.OPENAI:
        validator = TypeAdapter(ChatCompletionToolParam)
    else:
        validator = TypeAdapter(anthropic.types.ToolParam)

    validated_params = []
    for param in params:
        validated = validator.validate_python(param)
        validated_params.append(validated)

    return validated_params


class TestToolkit:
    """Test Toolkit class functionality."""

    def test_toolkit_name(self):
        """Test that toolkit name returns class name."""
        toolkit = SystemInfoToolkit()
        assert toolkit.name == "SystemInfoToolkit"

    def test_get_tools_finds_decorated_methods(self):
        """Test that get_tools finds all @tool_def decorated methods."""
        toolkit = SystemInfoToolkit()
        tools = toolkit.get_tools()

        tool_names = [tool.name for tool in tools]
        expected_tools = [
            "get_system_architecture",
            "get_timezone",
            "get_platform_info",
        ]

        assert len(tools) == 3
        for expected_tool in expected_tools:
            assert expected_tool in tool_names

    def test_tool_names_property(self):
        """Test tool_names property returns list of tool names."""
        toolkit = SystemInfoToolkit()
        tool_names = toolkit.tool_names

        expected_names = [
            "get_system_architecture",
            "get_timezone",
            "get_platform_info",
        ]
        assert len(tool_names) == 3
        for expected_name in expected_names:
            assert expected_name in tool_names

    @pytest.mark.parametrize(
        "schema_type", [ToolParamSchema.OPENAI, ToolParamSchema.ANTHROPIC]
    )
    def test_as_param_schema_validation(self, schema_type):
        """Test schema generation for different providers."""
        toolkit = SystemInfoToolkit()
        validated_params = validate_toolkit_schema(toolkit, schema_type, 3)

        # Verify all tools have valid schemas
        assert len(validated_params) == 3

        # Check specific tool exists in schemas
        if schema_type == ToolParamSchema.OPENAI:
            tool_names = [param["function"]["name"] for param in validated_params]
        else:
            tool_names = [param["name"] for param in validated_params]

        assert "get_system_architecture" in tool_names
        assert "get_timezone" in tool_names
        assert "get_platform_info" in tool_names

    def test_execute_member_tool_sync(self):
        """Test executing sync member tool."""
        toolkit = SystemInfoToolkit()

        # Test get_timezone (pure sync)
        result = toolkit.execute_tool("get_timezone", {})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_execute_member_tool_with_async_wrapper(self):
        """Test executing tool that uses asyncio.run internally."""
        toolkit = SystemInfoToolkit()

        # Test get_system_architecture (uses asyncio.run internally)
        result = toolkit.execute_tool("get_system_architecture", {})
        assert isinstance(result, str)
        valid_architectures = ["x86_64", "amd64", "arm64", "aarch64", "i386", "armv7l"]
        assert result in valid_architectures

    def test_execute_member_tool_with_parameters(self):
        """Test executing member tool with parameters."""
        toolkit = SystemInfoToolkit()

        # Test without optional parameter
        result = toolkit.execute_tool("get_platform_info", {})
        assert isinstance(result, dict)
        assert "system" in result
        assert "version" not in result

        # Test with optional parameter
        result = toolkit.execute_tool("get_platform_info", {"include_version": True})
        assert isinstance(result, dict)
        assert "system" in result
        assert "version" in result

    @pytest.mark.asyncio
    async def test_aexecute_member_tool(self):
        """Test async execution of member tools."""
        toolkit = SystemInfoToolkit()

        # Test async execution of sync tool
        result = await toolkit.aexecute_tool("get_timezone", {})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_execute_tool_not_found(self):
        """Test executing non-existent tool raises error."""
        toolkit = SystemInfoToolkit()

        with pytest.raises(
            ValueError, match="Tool with name nonexistent_tool not found"
        ):
            toolkit.execute_tool("nonexistent_tool", {})

    @pytest.mark.asyncio
    async def test_aexecute_tool_not_found(self):
        """Test async executing non-existent tool raises error."""
        toolkit = SystemInfoToolkit()

        with pytest.raises(
            ValueError, match="Tool with name nonexistent_tool not found"
        ):
            await toolkit.aexecute_tool("nonexistent_tool", {})


class TestToolkitAsync:
    """Test Toolkit async functionality."""

    @pytest.mark.asyncio
    async def test_async_toolkit_tools(self):
        """Test toolkit with async tools."""
        toolkit = AsyncToolkit()

        # Test async tool
        result = await toolkit.aexecute_tool("async_operation", {"delay": 0.001})
        assert isinstance(result, str)
        assert "Completed after 0.001s" in result

        # Test sync tool via async execution
        result = await toolkit.aexecute_tool("sync_operation", {"value": "test"})
        assert result == "Processed: test"

    def test_sync_execute_async_tool_raises_error(self):
        """Test that sync execute of async tool raises RuntimeError."""
        toolkit = AsyncToolkit()

        with pytest.raises(
            RuntimeError, match="asynchronous and cannot be executed synchronously"
        ):
            toolkit.execute_tool("async_operation", {"delay": 0.001})

    @pytest.mark.asyncio
    async def test_mixed_sync_async_execution(self):
        """Test executing both sync and async tools in mixed toolkit."""
        toolkit = AsyncToolkit()

        # Async execution should work for both
        sync_result = await toolkit.aexecute_tool(
            "sync_operation", {"value": "sync_test"}
        )
        async_result = await toolkit.aexecute_tool("async_operation", {"delay": 0.001})

        assert sync_result == "Processed: sync_test"
        assert "Completed after 0.001s" in async_result


class TestToolkitMCP:
    """Test Toolkit MCP server integration."""

    @pytest.mark.asyncio
    async def test_register_mcp(self):
        """Test registering toolkit tools with MCP server."""
        mcp = FastMCP("test-server")
        toolkit = SystemInfoToolkit()

        toolkit.register_mcp(mcp)

        # Should be able to find all registered tools by name
        expected_tools = [
            "get_system_architecture",
            "get_timezone",
            "get_platform_info",
        ]
        for tool_name in expected_tools:
            registered_tool = await mcp.get_tool(tool_name)
            assert registered_tool is not None
            assert registered_tool.name == tool_name

    @pytest.mark.asyncio
    async def test_create_mcp_server(self):
        """Test creating MCP server with toolkit tools."""
        toolkit = SystemInfoToolkit()

        mcp = toolkit.create_mcp_server("system-info-server")

        assert mcp is not None

        # Should have all toolkit tools registered
        expected_tools = [
            "get_system_architecture",
            "get_timezone",
            "get_platform_info",
        ]
        for tool_name in expected_tools:
            registered_tool = await mcp.get_tool(tool_name)
            assert registered_tool is not None
            assert registered_tool.name == tool_name
