import json
import pytest
import litellm
from pydantic import ValidationError

import agtk
from agtk.types import ToolParamSchema


PACKAGE_JSON_CONTENT = """{
  "name": "react-app",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.8.1"
  },
  "devDependencies": {
    "@types/react": "^18.2.43",
    "@types/react-dom": "^18.2.17",
    "@vitejs/plugin-react": "^4.2.1",
    "typescript": "^5.2.2",
    "vite": "^5.0.8"
  }
}"""

TEST_MODELS = [
    "gemini/gemini-2.5-flash",
    "anthropic/claude-3-5-haiku-20241022",
    "openai/gpt-4o-mini",
]


class FileToolkit(agtk.Toolkit):
    """Test toolkit with file creation tool that handles JSON content as strings."""

    @agtk.tool_def()
    def create_file(self, path: str, content: str) -> dict:
        """Creates a file with given path and content.

        Args:
            path: File path as string
            content: File content as string (can be JSON formatted)

        Returns:
            dict with creation status and metadata
        """
        return {
            "success": True,
            "path": path,
            "content_type": type(content).__name__,
            "content_length": len(content),
            "is_json": self._is_valid_json(content),
        }

    def _is_valid_json(self, content: str) -> bool:
        """Helper to check if content is valid JSON"""
        try:
            json.loads(content)
            return True
        except (json.JSONDecodeError, TypeError):
            return False


def create_file_tool():
    """Helper to create file toolkit and get create_file tool."""
    toolkit = FileToolkit()
    tools = toolkit.get_tools()
    return next(tool for tool in tools if tool.name == "create_file")


class TestCreateFileTool:
    """Test create_file tool with JSON string validation."""

    def test_create_file_with_package_json(self):
        """Test with the exact JSON from error log."""
        toolkit = FileToolkit()

        result = toolkit.execute_tool(
            "create_file", {"path": "package.json", "content": PACKAGE_JSON_CONTENT}
        )

        assert result["success"] is True
        assert result["path"] == "package.json"
        assert result["content_type"] == "str"
        assert result["is_json"] is True
        assert result["content_length"] > 0

    def test_create_file_with_json_string(self):
        """Test that JSON content stays as string during validation."""
        toolkit = FileToolkit()
        json_content = '{"test": "value", "number": 123}'

        result = toolkit.execute_tool(
            "create_file", {"path": "test.json", "content": json_content}
        )

        assert result["success"] is True
        assert result["path"] == "test.json"
        assert result["content_type"] == "str"
        assert result["is_json"] is True
        assert result["content_length"] == len(json_content)

    def test_create_file_with_plain_text(self):
        """Test with regular text content."""
        toolkit = FileToolkit()
        text_content = "This is plain text content"

        result = toolkit.execute_tool(
            "create_file",
            {"path": "readme.txt", "content": text_content},
        )

        assert result["success"] is True
        assert result["path"] == "readme.txt"
        assert result["content_type"] == "str"
        assert result["is_json"] is False
        assert result["content_length"] == len(text_content)

    def test_create_file_validation_errors(self):
        """Test parameter validation errors."""
        toolkit = FileToolkit()

        with pytest.raises(ValidationError):
            toolkit.execute_tool("create_file", {})

        with pytest.raises(ValidationError):
            toolkit.execute_tool("create_file", {"path": "test.txt"})

    def test_create_file_schema(self):
        """Test schema generation for different providers."""
        tool = create_file_tool()

        # OpenAI schema validation
        openai_param = tool.as_param(ToolParamSchema.OPENAI)
        assert openai_param["function"]["name"] == "create_file"

        openai_props = openai_param["function"]["parameters"]["properties"]
        assert "path" in openai_props
        assert "content" in openai_props
        assert openai_props["path"]["type"] == "string"
        assert openai_props["content"]["type"] == "string"

        # Required parameters
        required = openai_param["function"]["parameters"].get("required", [])
        assert "path" in required
        assert "content" in required

        # Anthropic schema validation
        anthropic_param = tool.as_param(ToolParamSchema.ANTHROPIC)
        assert anthropic_param["name"] == "create_file"

        anthropic_props = anthropic_param["input_schema"]["properties"]
        assert "path" in anthropic_props
        assert "content" in anthropic_props
        assert anthropic_props["path"]["type"] == "string"
        assert anthropic_props["content"]["type"] == "string"

        # Required parameters
        required = anthropic_param["input_schema"].get("required", [])
        assert "path" in required
        assert "content" in required


@pytest.mark.integration
class TestCreateFileToolLiteLLM:
    """Integration tests using LiteLLM with multiple models."""

    @pytest.mark.parametrize("model_id", TEST_MODELS)
    def test_create_file_tool_calling_with_llm(self, model_id):
        """Test end-to-end tool calling with LiteLLM."""
        toolkit = FileToolkit()

        # Use LiteLLM to call the create_file tool
        response = litellm.completion(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": "Create a package.json file for a React project with TypeScript. Use the tool I give you to create the file.",
                }
            ],
            tools=toolkit.as_param(ToolParamSchema.OPENAI),
            tool_choice="auto",
        )

        # Verify response exists
        assert response.choices[0].message is not None

        # Check if tool was called
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]

            # Verify it's our create_file tool
            assert tool_call.function.name == "create_file"

            print(f"\n{tool_call.function.arguments=}\n")

            # Parse and validate arguments
            arguments = json.loads(tool_call.function.arguments)
            assert "path" in arguments, "Tool call must include path argument"
            assert "content" in arguments, "Tool call must include content argument"
            assert isinstance(arguments["path"], str), "Path argument must be string"
            assert isinstance(arguments["content"], str), (
                "Content argument must be string"
            )
            assert len(arguments["content"]) > 0, "Content should not be empty"

            # Execute tool and verify string handling
            result = toolkit.execute_tool("create_file", arguments)
            assert result["success"] is True
            assert result["content_type"] == "str", (
                f"Content must stay as string, got {result['content_type']}"
            )
            assert result["path"] == arguments["path"]
            assert result["content_length"] == len(arguments["content"])

            # Verify content is reasonable (likely JSON for this prompt)
            try:
                parsed_content = json.loads(arguments["content"])
                assert isinstance(parsed_content, dict), (
                    "Generated content should be JSON object"
                )
                assert result["is_json"] is True
            except json.JSONDecodeError:
                # Some models might return non-JSON content, which is acceptable
                pass
        else:
            pytest.fail(f"Model {model_id} did not call the tool")
