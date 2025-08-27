import asyncio
import logging
import platform
import time
from typing import Annotated, List, Dict, Optional, Any

import pytest
from pydantic import Field, ValidationError, TypeAdapter
import anthropic
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from fastmcp import FastMCP
import ltls
from ltls.types import Tool, ToolParamSchema

logger = logging.getLogger(__name__)


# Test tool functions
@ltls.tool_def()
def sync_string_tool() -> str:
    """Returns system architecture as string."""
    return platform.machine()


@ltls.tool_def()
async def async_string_tool() -> str:
    """Returns timezone name asynchronously."""
    await asyncio.sleep(0.001)
    return time.tzname[0]


@ltls.tool_def()
def sync_number_tool(x: int, y: float) -> float:
    """Multiplies integer and float numbers."""
    return x * y


@ltls.tool_def()
async def async_number_tool(x: int, y: float) -> float:
    """Adds integer and float numbers asynchronously."""
    await asyncio.sleep(0.001)
    return x + y


@ltls.tool_def()
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

        @ltls.tool_def(name="decorated_test", description="A test function")
        def test_function():
            pass

        assert hasattr(test_function, "_tool_def")
        tool_def_attr = getattr(test_function, "_tool_def")
        assert tool_def_attr.name == "decorated_test"
        assert tool_def_attr.description == "A test function"

    def test_tool_def_uses_docstring(self):
        """Test that tool_def uses docstring if no description provided."""

        @ltls.tool_def(name="docstring_test")
        def test_function():
            """This is the docstring."""
            pass

        assert hasattr(test_function, "_tool_def")
        tool_def_attr = getattr(test_function, "_tool_def")
        assert tool_def_attr.description == "This is the docstring."

    def test_tool_def_empty_description(self):
        """Test that tool_def sets empty description if neither provided."""

        @ltls.tool_def(name="empty_test")
        def test_function():
            pass

        assert hasattr(test_function, "_tool_def")
        tool_def_attr = getattr(test_function, "_tool_def")
        assert tool_def_attr.description == ""


class TestJSONParameterConversion:
    """Test JSON string parameter conversion for MCP compatibility.

    These tests ensure that when MCP clients send array/dict parameters
    as JSON strings, they are automatically converted to Python types.
    """

    # Test functions with various parameter types
    @staticmethod
    @ltls.tool_def()
    def process_list(items: List[str]) -> dict:
        """Process a list of strings."""
        return {"count": len(items), "items": items}

    @staticmethod
    @ltls.tool_def()
    def process_dict(data: Dict[str, Any]) -> dict:
        """Process a dictionary."""
        return {"keys": list(data.keys()), "data": data}

    @staticmethod
    @ltls.tool_def()
    def process_optional_list(items: Optional[List[int]] = None) -> dict:
        """Process an optional list of integers."""
        if items is None:
            return {"items": None}
        return {"sum": sum(items), "items": items}

    @staticmethod
    @ltls.tool_def()
    def process_string(content: str) -> dict:
        """Process a string that might contain JSON."""
        return {"content": content, "length": len(content)}

    @staticmethod
    @ltls.tool_def()
    def process_mixed(
        text: str,
        numbers: List[float],
        config: Dict[str, str],
        optional_tags: Optional[List[str]] = None,
    ) -> dict:
        """Process mixed parameter types."""
        return {
            "text": text,
            "numbers": numbers,
            "config": config,
            "tags": optional_tags,
        }

    @staticmethod
    @ltls.tool_def()
    def process_nested_list(matrix: List[List[int]]) -> dict:
        """Process nested list structure."""
        return {
            "rows": len(matrix),
            "cols": len(matrix[0]) if matrix else 0,
            "matrix": matrix,
        }

    def test_list_parameter_with_json_string(self):
        """Test that JSON string arrays are converted to Python lists."""
        tool = create_tool(self.process_list)

        # Test with JSON string (MCP client behavior)
        result = tool.execute({"items": '["apple", "banana", "cherry"]'})
        assert result["count"] == 3
        assert result["items"] == ["apple", "banana", "cherry"]

        # Test with native list (should still work)
        result = tool.execute({"items": ["apple", "banana", "cherry"]})
        assert result["count"] == 3
        assert result["items"] == ["apple", "banana", "cherry"]

    def test_dict_parameter_with_json_string(self):
        """Test that JSON string objects are converted to Python dicts."""
        tool = create_tool(self.process_dict)

        # Test with JSON string
        result = tool.execute({"data": '{"name": "John", "age": 30}'})
        assert "name" in result["keys"]
        assert "age" in result["keys"]
        assert result["data"]["name"] == "John"
        assert result["data"]["age"] == 30

        # Test with native dict
        result = tool.execute({"data": {"name": "John", "age": 30}})
        assert result["data"]["name"] == "John"

    def test_optional_list_with_json_string(self):
        """Test Optional[List] parameters with JSON strings."""
        tool = create_tool(self.process_optional_list)

        # Test with JSON string
        result = tool.execute({"items": "[1, 2, 3, 4, 5]"})
        assert result["sum"] == 15
        assert result["items"] == [1, 2, 3, 4, 5]

        # Test with None (omitted parameter)
        result = tool.execute({})
        assert result["items"] is None

        # Test with native list
        result = tool.execute({"items": [1, 2, 3]})
        assert result["sum"] == 6

    def test_string_parameter_preserves_json(self):
        """Test that string parameters containing JSON are NOT parsed."""
        tool = create_tool(self.process_string)

        # JSON object string should stay as string
        json_obj = '{"this": "should remain a string"}'
        result = tool.execute({"content": json_obj})
        assert result["content"] == json_obj
        assert isinstance(result["content"], str)

        # JSON array string should also stay as string
        json_arr = '["this", "should", "also", "remain", "a", "string"]'
        result = tool.execute({"content": json_arr})
        assert result["content"] == json_arr
        assert isinstance(result["content"], str)

    def test_mixed_parameters_with_json_strings(self):
        """Test mixed parameter types with some as JSON strings."""
        tool = create_tool(self.process_mixed)

        result = tool.execute(
            {
                "text": "Hello World",
                "numbers": "[1.5, 2.5, 3.5]",
                "config": '{"env": "prod", "debug": "false"}',
                "optional_tags": '["urgent", "feature"]',
            }
        )

        assert result["text"] == "Hello World"
        assert result["numbers"] == [1.5, 2.5, 3.5]
        assert result["config"]["env"] == "prod"
        assert result["config"]["debug"] == "false"
        assert result["tags"] == ["urgent", "feature"]

    def test_invalid_json_stays_as_string(self):
        """Test that invalid JSON strings cause validation errors."""
        tool = create_tool(self.process_list)

        # Invalid JSON should cause validation error
        with pytest.raises(ValidationError) as exc_info:
            tool.execute({"items": "[not valid json"})

        error = exc_info.value
        assert "list" in str(error).lower()

    def test_empty_collections_as_json_strings(self):
        """Test empty arrays and objects as JSON strings."""
        list_tool = create_tool(self.process_list)
        dict_tool = create_tool(self.process_dict)

        # Empty array
        result = list_tool.execute({"items": "[]"})
        assert result["count"] == 0
        assert result["items"] == []

        # Empty object
        result = dict_tool.execute({"data": "{}"})
        assert result["keys"] == []
        assert result["data"] == {}

    def test_nested_structures_as_json_strings(self):
        """Test nested data structures as JSON strings."""
        tool = create_tool(self.process_nested_list)

        # Nested list as JSON string
        result = tool.execute({"matrix": "[[1, 2, 3], [4, 5, 6], [7, 8, 9]]"})
        assert result["rows"] == 3
        assert result["cols"] == 3
        assert result["matrix"][1][1] == 5

        # Native nested list should still work
        result = tool.execute({"matrix": [[1, 2], [3, 4]]})
        assert result["rows"] == 2
        assert result["cols"] == 2

    def test_json_with_whitespace(self):
        """Test JSON strings with various whitespace formatting."""
        tool = create_tool(self.process_list)

        # JSON with extra whitespace
        result = tool.execute({"items": '  [  "a" , "b" , "c"  ]  '})
        assert result["items"] == ["a", "b", "c"]

        # JSON with newlines and tabs
        result = tool.execute({"items": '[\n\t"x",\n\t"y"\n]'})
        assert result["items"] == ["x", "y"]

    def test_type_mismatch_after_parsing(self):
        """Test that type validation still works after JSON parsing."""
        tool = create_tool(self.process_list)

        # JSON parses but wrong type (object instead of array)
        with pytest.raises(ValidationError):
            tool.execute({"items": '{"not": "an array"}'})

        # JSON array but wrong element types
        int_tool = create_tool(self.process_optional_list)
        with pytest.raises(ValidationError):
            int_tool.execute({"items": '["not", "integers"]'})

    @pytest.mark.asyncio
    async def test_async_execution_with_json_strings(self):
        """Test async execution also handles JSON string conversion."""
        tool = create_tool(self.process_list)

        # Async execution with JSON string
        result = await tool.aexecute({"items": '["async", "test"]'})
        assert result["count"] == 2
        assert result["items"] == ["async", "test"]

    def test_regression_without_fix(self):
        """Test that would fail if JSON preprocessing is removed.

        This test documents the exact MCP client behavior that requires
        the JSON string preprocessing fix in Tool._preprocess_json_arguments.
        """

        # This is the exact format MCP clients send
        @ltls.tool_def()
        def mcp_style_function(group_by: List[str]) -> dict:
            """Function expecting list parameter like in MCP tools."""
            return {"grouped_by": group_by}

        tool = create_tool(mcp_style_function)

        # This is what MCP clients send - JSON string instead of list
        mcp_style_input = {"group_by": '["type", "status"]'}

        # This should work with our fix
        result = tool.execute(mcp_style_input)
        assert result["grouped_by"] == ["type", "status"]

        # Document the error that would occur without the fix:
        # ValidationError: Input should be a valid list [type=list_type, input_value='["type", "status"]', input_type=str]
