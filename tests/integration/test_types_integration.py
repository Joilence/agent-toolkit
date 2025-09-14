import os
import pytest
import anthropic
from typing import Union

from agtk.types import (
    Toolkit,
    ToolkitSuite,
    ToolParamSchema,
    tool_def,
)


# Test toolkit for integration tests
class TestIntegrationToolkit(Toolkit):
    """A test toolkit for integration testing."""

    @tool_def(name="echo", description="Echo back the input")
    def echo(self, message: str) -> str:
        """Echo back the input message."""
        return f"You said: {message}"

    @tool_def(name="add", description="Add two numbers")
    def add(self, a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """Add two numbers and return the result."""
        return a + b


# Another test toolkit for testing ToolkitSuite
class AnotherIntegrationToolkit(Toolkit):
    """Another test toolkit for integration testing."""

    @tool_def(name="multiply", description="Multiply two numbers")
    def multiply(self, a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """Multiply two numbers and return the result."""
        return a * b


@pytest.mark.integration
class TestToolkitIntegration:
    """Integration tests for Toolkit with real LLM providers."""

    def setup_method(self):
        """Set up the test by creating a test toolkit."""
        self.toolkit = TestIntegrationToolkit()

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY environment variable not set",
    )
    def test_toolkit_with_anthropic(self):
        """Test Toolkit integration with Anthropic Claude."""
        # Convert toolkit to Anthropic tool parameters
        tools = self.toolkit.as_param(ToolParamSchema.ANTHROPIC)

        # Verify the tool parameters format
        assert isinstance(tools, list)
        assert len(tools) == 2  # echo and add

        # Check the format of each tool
        for tool in tools:
            assert isinstance(tool, dict)
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool

        # Only test API call if we have an API key
        if os.environ.get("ANTHROPIC_API_KEY"):
            # Create Anthropic client
            client = anthropic.Client()

            # Make API call
            response = client.messages.create(
                model="claude-3-haiku-20240307",  # Using the smallest/fastest model for testing
                messages=[
                    {"role": "user", "content": "What tools do you have available?"}
                ],
                tools=tools,
                max_tokens=1024,
                temperature=0,
            )

            # Assert the response contains mentions of our tools
            response_text = " ".join(
                [block.text for block in response.content if hasattr(block, "text")]
            )
            # Check that all tools are mentioned in the response
            for tool_name in ["echo", "add"]:
                assert tool_name in response_text.lower(), (
                    f"Tool '{tool_name}' not mentioned in response"
                )

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable not set",
    )
    def test_toolkit_with_openai(self):
        """Test Toolkit integration with OpenAI."""
        try:
            import openai
        except ImportError:
            pytest.skip("openai package not installed")

        # Convert toolkit to OpenAI tool parameters
        tools = self.toolkit.as_param(ToolParamSchema.OPENAI)

        # Verify the tool parameters format
        assert isinstance(tools, list)
        assert len(tools) == 2  # echo and add

        # Check the format of each tool
        for tool in tools:
            # Tools could be either dict format or OpenAIToolParam objects based on implementation
            if hasattr(tool, "type"):
                # Object style
                assert tool.type == "function"
                assert hasattr(tool, "function")
                assert hasattr(tool.function, "name")
            else:
                # Dict style
                assert isinstance(tool, dict)
                assert tool.get("type") == "function"
                assert "function" in tool
                assert "name" in tool["function"] or hasattr(tool["function"], "name")

        # Only test API call if we have an API key
        if os.environ.get("OPENAI_API_KEY"):
            # Create OpenAI client
            client = openai.OpenAI()

            # Make API call
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using cheaper model for testing
                messages=[
                    {"role": "user", "content": "What tools do you have available?"}
                ],
                tools=tools,
                max_tokens=1024,
                temperature=0,
            )

            # Verify the response contains mentions of our tools
            assert response.choices[0].message is not None
            response_text = response.choices[0].message.content or ""
            for tool_name in ["echo", "add"]:
                assert tool_name in response_text.lower(), (
                    f"Tool '{tool_name}' not mentioned in response"
                )


@pytest.mark.integration
class TestToolkitSuiteIntegration:
    """Integration tests for ToolkitSuite with real LLM providers."""

    def setup_method(self):
        """Set up the test by creating a test toolkit suite."""
        self.toolkit1 = TestIntegrationToolkit()
        self.toolkit2 = AnotherIntegrationToolkit()
        self.suite = ToolkitSuite([self.toolkit1, self.toolkit2])

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY environment variable not set",
    )
    def test_toolkit_suite_with_anthropic(self):
        """Test ToolkitSuite integration with Anthropic Claude."""
        # Convert suite to Anthropic tool parameters
        tools = self.suite.as_param(ToolParamSchema.ANTHROPIC)

        # Verify the tool parameters format
        assert isinstance(tools, list)
        assert len(tools) == 3  # echo, add, and multiply

        # Only test API call if we have an API key
        if os.environ.get("ANTHROPIC_API_KEY"):
            # Create Anthropic client
            client = anthropic.Client()

            # Make API call
            response = client.messages.create(
                model="claude-3-haiku-20240307",  # Using the smallest/fastest model for testing
                messages=[
                    {"role": "user", "content": "What tools do you have available?"}
                ],
                tools=tools,
                max_tokens=1024,
                temperature=0,
            )

            # Assert the response contains mentions of our tools
            response_text = " ".join(
                [block.text for block in response.content if hasattr(block, "text")]
            )
            # Check that all tools from each toolkit are mentioned
            for tool_name in ["echo", "add", "multiply"]:
                assert tool_name in response_text.lower(), (
                    f"Tool '{tool_name}' not mentioned in response"
                )

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable not set",
    )
    def test_toolkit_suite_with_openai(self):
        """Test ToolkitSuite integration with OpenAI."""
        try:
            import openai
        except ImportError:
            pytest.skip("openai package not installed")

        # Convert suite to OpenAI tool parameters
        tools = self.suite.as_param(ToolParamSchema.OPENAI)

        # Verify the tool parameters format
        assert isinstance(tools, list)
        assert len(tools) == 3  # echo, add, and multiply

        # Only test API call if we have an API key
        if os.environ.get("OPENAI_API_KEY"):
            # Create OpenAI client
            client = openai.OpenAI()

            # Make API call
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using cheaper model for testing
                messages=[
                    {"role": "user", "content": "What tools do you have available?"}
                ],
                tools=tools,
                max_tokens=1024,
                temperature=0,
            )

            # Verify the response contains mentions of our tools
            assert response.choices[0].message is not None
            response_text = response.choices[0].message.content or ""
            # Not all OpenAI models consistently mention tools, but we'll check anyway
            # and provide more context if the assertion fails
            for tool_name in ["echo", "add", "multiply"]:
                assert tool_name in response_text.lower(), (
                    f"Tool '{tool_name}' not mentioned in response"
                )


@pytest.mark.integration
class TestToolExecutionIntegration:
    """Integration tests for Tool execution."""

    def test_execute_tool(self):
        """Test that a tool can be executed through ToolkitSuite."""
        # Setup
        toolkit = TestIntegrationToolkit()
        suite = ToolkitSuite([toolkit])

        # Execute the add tool
        result = suite.execute_tool("add", {"a": 2, "b": 3})

        # Verify
        assert result == 5

        # Execute the echo tool
        result = suite.execute_tool("echo", {"message": "hello"})

        # Verify
        assert result == "You said: hello"


if __name__ == "__main__":
    # For manual testing, this section allows running the integration tests directly

    # Get all test classes
    test_classes = [TestToolkitIntegration, TestToolkitSuiteIntegration]

    for test_class in test_classes:
        print(f"\nRunning tests for {test_class.__name__}...")

        # Create an instance of the test class
        instance = test_class()

        # Run setup
        if hasattr(instance, "setup_method"):
            instance.setup_method()

        # Find and run all test methods
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                test_method = getattr(instance, method_name)

                # Skip tests that require API keys if they're not set
                if "anthropic" in method_name and not os.environ.get(
                    "ANTHROPIC_API_KEY"
                ):
                    print(f"  Skipping {method_name} (ANTHROPIC_API_KEY not set)")
                    continue

                if "openai" in method_name and not os.environ.get("OPENAI_API_KEY"):
                    print(f"  Skipping {method_name} (OPENAI_API_KEY not set)")
                    continue

                print(f"  Running {method_name}...")
                try:
                    test_method()
                    print(f"  ✓ {method_name} passed!")
                except Exception as e:
                    print(f"  ✗ {method_name} failed: {e}")

    # Also run the async test manually
    import asyncio

    print("\nRunning async tests...")
    async_test = TestToolExecutionIntegration()

    async def run_async_tests():
        try:
            await async_test.test_execute_tool()
            print("  ✓ test_execute_tool passed!")
        except Exception as e:
            print(f"  ✗ test_execute_tool failed: {e}")

    asyncio.run(run_async_tests())
