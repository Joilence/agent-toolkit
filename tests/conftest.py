import json
from typing import Annotated, Optional
import agtk
from pydantic import Field
import pytest


class TestToolkit(agtk.Toolkit):
    """A simple toolkit for testing."""

    @agtk.tool_def(name="tool_with_params")
    def tool_with_params(
        self,
        str_param: Annotated[str, Field(description="A required string parameter")],
        int_param: Annotated[
            Optional[int], Field(description="An optional int parameter")
        ] = None,
        bool_param: Annotated[
            bool,
            Field(
                description="An optional boolean parameter with default value to False",
            ),
        ] = False,
    ):
        """Annotated parameter tool from docstring"""
        return f"Executed with {str_param}, {int_param} and {bool_param}"


@agtk.tool_def()
def external_tool_function(
    str_param: Annotated[str, Field(description="A required string parameter")],
    int_param: Annotated[
        Optional[int], Field(description="An optional int parameter")
    ] = None,
    bool_param: Annotated[
        bool,
        Field(
            description="An optional boolean parameter with default value to False",
        ),
    ] = False,
):
    """External tool with annotated parameters"""
    return f"Executed external tool with {str_param}, {int_param} and {bool_param}"


@pytest.fixture
def test_toolkit():
    return TestToolkit()


if __name__ == "__main__":
    """Test output of param by running `uv run -- python3 tests/conftest.py`"""
    toolkit = TestToolkit()
    print(json.dumps(toolkit.as_param(agtk.ToolParamSchema.OPENAI), indent=2))
