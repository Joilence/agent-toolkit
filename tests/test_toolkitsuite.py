import asyncio
import logging
import platform
import time

import pytest
from pydantic import TypeAdapter
import anthropic
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from fastmcp import FastMCP
import agtk
from agtk.types import ToolParamSchema, ToolkitSuite

logger = logging.getLogger(__name__)


# Test toolkits for suite testing
class SystemInfoToolkit(agtk.Toolkit):
    """System information toolkit."""

    @agtk.tool_def()
    def get_system_architecture(self):
        """Gets system architecture."""
        return platform.machine()

    @agtk.tool_def()
    def get_timezone(self):
        """Gets timezone information."""
        return time.tzname[0]


class MathToolkit(agtk.Toolkit):
    """Mathematical operations toolkit."""

    @agtk.tool_def()
    def add_numbers(self, x: int, y: int) -> int:
        """Adds two numbers."""
        return x + y

    @agtk.tool_def()
    def multiply_numbers(self, x: float, y: float) -> float:
        """Multiplies two numbers."""
        return x * y


class AsyncToolkit(agtk.Toolkit):
    """Async operations toolkit."""

    @agtk.tool_def()
    async def async_operation(self, delay: float = 0.001) -> str:
        """Performs async operation with delay."""
        await asyncio.sleep(delay)
        return f"Completed after {delay}s"

    @agtk.tool_def()
    def sync_operation(self, value: str) -> str:
        """Performs sync operation."""
        return f"Processed: {value}"


# Helper functions
def create_test_suite(*toolkits):
    """Helper to create ToolkitSuite with given toolkits."""
    return ToolkitSuite(list(toolkits))


def validate_suite_schema(suite, schema_type, expected_tool_count=None):
    """Helper to validate suite schema with TypeAdapter."""
    params = suite.as_param(schema_type)

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


class TestToolkitSuite:
    """Test ToolkitSuite class functionality."""

    def test_init_empty(self):
        """Test initialization with no toolkits."""
        suite = ToolkitSuite()
        assert suite.get_toolkits() == []
        assert suite.tool_names == []
        assert suite.get_tools() == []

    def test_init_with_toolkits(self):
        """Test initialization with toolkits."""
        system_toolkit = SystemInfoToolkit()
        math_toolkit = MathToolkit()

        suite = ToolkitSuite([system_toolkit, math_toolkit])

        toolkits = suite.get_toolkits()
        assert len(toolkits) == 2
        assert system_toolkit in toolkits
        assert math_toolkit in toolkits

    def test_tool_names(self):
        """Test tool_names property aggregates from all toolkits."""
        suite = create_test_suite(SystemInfoToolkit(), MathToolkit())

        tool_names = suite.tool_names
        expected_names = [
            "get_system_architecture",
            "get_timezone",
            "add_numbers",
            "multiply_numbers",
        ]

        assert len(tool_names) == 4
        for expected_name in expected_names:
            assert expected_name in tool_names

    @pytest.mark.parametrize(
        "schema_type", [ToolParamSchema.OPENAI, ToolParamSchema.ANTHROPIC]
    )
    def test_as_param_schema_validation(self, schema_type):
        """Test schema generation for different providers."""
        suite = create_test_suite(SystemInfoToolkit(), MathToolkit())
        validated_params = validate_suite_schema(suite, schema_type, 4)

        # Verify all tools have valid schemas
        assert len(validated_params) == 4

        # Check specific tools exist in schemas
        if schema_type == ToolParamSchema.OPENAI:
            tool_names = [param["function"]["name"] for param in validated_params]
        else:
            tool_names = [param["name"] for param in validated_params]

        expected_tools = [
            "get_system_architecture",
            "get_timezone",
            "add_numbers",
            "multiply_numbers",
        ]
        for expected_tool in expected_tools:
            assert expected_tool in tool_names

    def test_add_toolkit(self):
        """Test adding toolkit to suite."""
        suite = ToolkitSuite()
        system_toolkit = SystemInfoToolkit()

        suite.add_toolkit(system_toolkit)

        toolkits = suite.get_toolkits()
        assert len(toolkits) == 1
        assert system_toolkit in toolkits

        # Verify tools are accessible
        tool_names = suite.tool_names
        assert "get_system_architecture" in tool_names
        assert "get_timezone" in tool_names

    def test_get_toolkits(self):
        """Test getting all toolkits from suite."""
        system_toolkit = SystemInfoToolkit()
        math_toolkit = MathToolkit()
        suite = create_test_suite(system_toolkit, math_toolkit)

        toolkits = suite.get_toolkits()
        assert len(toolkits) == 2
        assert system_toolkit in toolkits
        assert math_toolkit in toolkits

    def test_get_tools(self):
        """Test getting all tools from all toolkits."""
        suite = create_test_suite(SystemInfoToolkit(), MathToolkit())

        tools = suite.get_tools()
        tool_names = [tool.name for tool in tools]

        expected_tools = [
            "get_system_architecture",
            "get_timezone",
            "add_numbers",
            "multiply_numbers",
        ]

        assert len(tools) == 4
        for expected_tool in expected_tools:
            assert expected_tool in tool_names

    def test_execute_tool_sync(self):
        """Test synchronous tool execution from suite."""
        suite = create_test_suite(SystemInfoToolkit(), MathToolkit())

        # Test system tool execution
        result = suite.execute_tool("get_system_architecture", {})
        assert isinstance(result, str)
        valid_architectures = ["x86_64", "amd64", "arm64", "aarch64", "i386", "armv7l"]
        assert result in valid_architectures

        # Test math tool execution
        result = suite.execute_tool("add_numbers", {"x": 5, "y": 3})
        assert result == 8

        result = suite.execute_tool("multiply_numbers", {"x": 2.5, "y": 4.0})
        assert result == 10.0

    @pytest.mark.asyncio
    async def test_aexecute_tool_async(self):
        """Test asynchronous tool execution from suite."""
        suite = create_test_suite(SystemInfoToolkit(), AsyncToolkit())

        # Test sync tool via async execution
        result = await suite.aexecute_tool("get_timezone", {})
        assert isinstance(result, str)
        assert len(result) > 0

        # Test async tool execution
        result = await suite.aexecute_tool("async_operation", {"delay": 0.001})
        assert "Completed after 0.001s" in result

        # Test sync tool from async toolkit
        result = await suite.aexecute_tool("sync_operation", {"value": "test"})
        assert result == "Processed: test"

    def test_execute_tool_not_found(self):
        """Test executing non-existent tool raises error."""
        suite = create_test_suite(SystemInfoToolkit())

        with pytest.raises(
            ValueError, match="Tool with name nonexistent_tool not found"
        ):
            suite.execute_tool("nonexistent_tool", {})

    @pytest.mark.asyncio
    async def test_aexecute_tool_not_found(self):
        """Test async executing non-existent tool raises error."""
        suite = create_test_suite(SystemInfoToolkit())

        with pytest.raises(
            ValueError, match="Tool with name nonexistent_tool not found"
        ):
            await suite.aexecute_tool("nonexistent_tool", {})

    def test_sync_execute_async_tool_raises_error(self):
        """Test that sync execute of async tool raises RuntimeError."""
        suite = create_test_suite(AsyncToolkit())

        with pytest.raises(
            RuntimeError, match="asynchronous and cannot be executed synchronously"
        ):
            suite.execute_tool("async_operation", {"delay": 0.001})

    @pytest.mark.asyncio
    async def test_register_mcp(self):
        """Test registering suite tools with MCP server."""
        mcp = FastMCP("test-suite-server")
        suite = create_test_suite(SystemInfoToolkit(), MathToolkit())

        suite.register_mcp(mcp)

        # Should be able to find all registered tools by name
        expected_tools = [
            "get_system_architecture",
            "get_timezone",
            "add_numbers",
            "multiply_numbers",
        ]
        for tool_name in expected_tools:
            registered_tool = await mcp.get_tool(tool_name)
            assert registered_tool is not None
            assert registered_tool.name == tool_name

    @pytest.mark.asyncio
    async def test_mixed_toolkit_coordination(self):
        """Test coordination between different types of toolkits."""
        suite = create_test_suite(SystemInfoToolkit(), MathToolkit(), AsyncToolkit())

        # Execute tools from different toolkits in sequence
        arch_result = await suite.aexecute_tool("get_system_architecture", {})
        math_result = await suite.aexecute_tool("add_numbers", {"x": 10, "y": 20})
        async_result = await suite.aexecute_tool("async_operation", {"delay": 0.001})

        assert isinstance(arch_result, str)
        assert math_result == 30
        assert "Completed after 0.001s" in async_result

        # Verify all tools are accessible
        all_tools = suite.get_tools()
        assert len(all_tools) == 6  # 2 + 2 + 2 tools from the toolkits

        tool_names = [tool.name for tool in all_tools]
        expected_names = [
            "get_system_architecture",
            "get_timezone",  # SystemInfoToolkit
            "add_numbers",
            "multiply_numbers",  # MathToolkit
            "async_operation",
            "sync_operation",  # AsyncToolkit
        ]
        for expected_name in expected_names:
            assert expected_name in tool_names
