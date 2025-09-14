import asyncio
import logging
import platform
import time
from typing import Annotated

import pytest
from pydantic import Field, ValidationError, TypeAdapter
import anthropic
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from fastmcp import FastMCP
import agtk
from agtk.types import Tool, ToolParamSchema

logger = logging.getLogger(__name__)


# Test tool functions
@agtk.tool_def()
def sync_string_tool() -> str:
    """Returns system architecture as string."""
    return platform.machine()


@agtk.tool_def()
async def async_string_tool() -> str:
    """Returns timezone name asynchronously."""
    await asyncio.sleep(0.001)
    return time.tzname[0]


@agtk.tool_def()
def sync_number_tool(x: int, y: float) -> float:
    """Multiplies integer and float numbers."""
    return x * y


@agtk.tool_def()
async def async_number_tool(x: int, y: float) -> float:
    """Adds integer and float numbers asynchronously."""
    await asyncio.sleep(0.001)
    return x + y


@agtk.tool_def()
def sync_complex_tool(
    data: Annotated[dict, Field(description="Input data dictionary")],
    optional_param: Annotated[bool, Field(description="Optional boolean flag")] = True,
) -> dict:
    """Processes data dictionary with optional parameter."""
    return {"processed": data, "flag": optional_param}


# Helper functions
def create_tool(tool_fn):
    """Helper to create Tool from decorated function."""
    return Tool.from_function(
        fn=tool_fn,
        name=tool_fn._tool_def.name,
        description=tool_fn._tool_def.description,
    )


def validate_api_schema(tool, schema_type, expected_count=None):
    """Helper to validate schema with TypeAdapter."""
    if schema_type == ToolParamSchema.OPENAI:
        validator = TypeAdapter(ChatCompletionToolParam)
        param = tool.as_param(ToolParamSchema.OPENAI)
    else:
        validator = TypeAdapter(anthropic.types.ToolParam)
        param = tool.as_param(ToolParamSchema.ANTHROPIC)

    validated = validator.validate_python(param)

    if expected_count is not None:
        if schema_type == ToolParamSchema.OPENAI:
            count = len(validated["function"]["parameters"].get("properties", {}))
        else:
            count = len(validated["input_schema"].get("properties", {}))
        assert count == expected_count

    return validated


class TestTool:
    """Test Tool class creation and schema generation."""

    def test_from_function_creation(self):
        """Test creating Tool from decorated function."""
        tool = create_tool(async_string_tool)
        assert tool.name == "async_string_tool"
        assert tool.description == "Returns timezone name asynchronously."

    def test_schema_validation(self):
        """Test both OpenAI and Anthropic schema validation."""
        tool = create_tool(sync_complex_tool)

        # Validate both schemas with parameter count
        openai_param = validate_api_schema(tool, ToolParamSchema.OPENAI, 2)
        anthropic_param = validate_api_schema(tool, ToolParamSchema.ANTHROPIC, 2)

        # Basic structure validation
        assert openai_param["function"]["name"] == "sync_complex_tool"
        assert anthropic_param["name"] == "sync_complex_tool"

    @pytest.mark.parametrize(
        "tool_fn,is_async", [(sync_string_tool, False), (async_string_tool, True)]
    )
    @pytest.mark.asyncio
    async def test_tool_execution(self, tool_fn, is_async):
        """Test both sync and async tool execution."""
        tool = create_tool(tool_fn)

        # Test .run() method
        result = await tool.run({})
        assert result is not None

        # Test execute/aexecute methods
        if is_async:
            result = await tool.aexecute({})
        else:
            result = tool.execute({})

        assert isinstance(result, str)
        assert len(result) > 0

        if is_async:
            # Test execute should raise RuntimeError if tool is async
            with pytest.raises(RuntimeError):
                tool.execute({})
        else:
            # Test aexecute should execute sync tool
            result = await tool.aexecute({})
            assert result is not None

    @pytest.mark.parametrize(
        "tool_fn,is_async", [(sync_number_tool, False), (async_number_tool, True)]
    )
    @pytest.mark.asyncio
    async def test_validation_errors(self, tool_fn, is_async):
        """Test validation errors for both sync and async tools."""
        tool = create_tool(tool_fn)
        execute_fn = tool.aexecute if is_async else tool.execute

        # Test missing arguments
        with pytest.raises(ValidationError) as exc_info:
            if is_async:
                await execute_fn({})
            else:
                execute_fn({})

        error = exc_info.value
        assert len(error.errors()) == 2  # x and y are both missing
        error_dict = {err["loc"][0]: err for err in error.errors()}
        assert "x" in error_dict and "y" in error_dict
        assert all(err["type"] == "missing_argument" for err in error_dict.values())

        # Test wrong argument types
        with pytest.raises(ValidationError) as exc_info:
            if is_async:
                await execute_fn({"x": "not_a_number", "y": "also_not_a_number"})
            else:
                execute_fn({"x": "not_a_number", "y": "also_not_a_number"})

        error = exc_info.value
        assert len(error.errors()) == 2
        error_dict = {err["loc"][0]: err for err in error.errors()}
        assert error_dict["x"]["type"] == "int_parsing"
        assert error_dict["y"]["type"] == "float_parsing"

    def test_complex_tool_validation_errors(self):
        """Test validation errors for complex tool."""
        tool = create_tool(sync_complex_tool)

        with pytest.raises(ValidationError, match="required|missing"):
            tool.execute({})

        with pytest.raises(ValidationError, match="dict|type|invalid"):
            tool.execute({"data": "not_a_dict"})

        with pytest.raises(ValidationError, match="bool|type|invalid"):
            tool.execute({"data": {"key": "value"}, "optional_param": "not_a_bool"})

    @pytest.mark.parametrize(
        "tool_fn,expected_count",
        [
            (sync_string_tool, 0),
            (async_string_tool, 0),
            (sync_number_tool, 2),
            (sync_complex_tool, 2),
        ],
    )
    def test_schema_inspection(self, tool_fn, expected_count):
        """Test schema inspection for different tools with API type validation."""
        tool = create_tool(tool_fn)

        # Validate both OpenAI and Anthropic schemas
        validate_api_schema(tool, ToolParamSchema.OPENAI, expected_count)
        validate_api_schema(tool, ToolParamSchema.ANTHROPIC, expected_count)


class TestFastMCPIntegration:
    """Test FastMCP server integration."""

    @pytest.mark.asyncio
    async def test_tool_to_mcp_registration(self):
        """Test registering tool with FastMCP server."""

        mcp = FastMCP("test-server")
        tool = create_tool(sync_string_tool)
        mcp.add_tool(tool)

        # Should find the tool by name
        registered_tool = await mcp.get_tool(tool.name)
        assert registered_tool is not None
        # Should have the correct tool registered
        assert registered_tool.name == tool.name


class TestToolDef:
    """Test @tool_def decorator functionality."""

    def test_tool_def_decorator(self):
        """Test that tool_def decorator attaches _tool_def attribute."""

        @agtk.tool_def(name="decorated_test", description="A test function")
        def test_function():
            pass

        assert hasattr(test_function, "_tool_def")
        tool_def_attr = getattr(test_function, "_tool_def")
        assert tool_def_attr.name == "decorated_test"
        assert tool_def_attr.description == "A test function"

    def test_tool_def_uses_docstring(self):
        """Test that tool_def uses docstring if no description provided."""

        @agtk.tool_def(name="docstring_test")
        def test_function():
            """This is the docstring."""
            pass

        assert hasattr(test_function, "_tool_def")
        tool_def_attr = getattr(test_function, "_tool_def")
        assert tool_def_attr.description == "This is the docstring."

    def test_tool_def_empty_description(self):
        """Test that tool_def sets empty description if neither provided."""

        @agtk.tool_def(name="empty_test")
        def test_function():
            pass

        assert hasattr(test_function, "_tool_def")
        tool_def_attr = getattr(test_function, "_tool_def")
        assert tool_def_attr.description == ""
