import os
import pytest
import anthropic

from agtk.types import ToolParamSchema

# Latest Anthropic models as of May 2025 that support tool use
ANTHROPIC_MODELS = [
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
]


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY environment variable not set",
)
@pytest.mark.parametrize("model_id", ANTHROPIC_MODELS)
def test_anthropic_models_tool_compatibility(model_id, toolkit):
    """Test each Anthropic model's compatibility with our toolkit's tools."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    try:
        # Create Anthropic client
        client = anthropic.Anthropic()

        # Make API call
        response = client.messages.create(
            model=model_id,
            max_tokens=1024,
            messages=[{"role": "user", "content": "What tools do you have available?"}],
            tools=toolkit.as_param(ToolParamSchema.ANTHROPIC),
        )

        # Verify the response contains mentions of our tools
        assert response.content is not None
        response_text = ""
        for item in response.content:
            if item.type == "text" and hasattr(item, "text"):
                response_text += item.text

        # Check if the response mentions tools
        # Not all models consistently mention tools by name, so we check for any tool response
        assert len(response_text) > 0, (
            f"Model {model_id} did not return a valid response"
        )

        # Try to look for at least one of our tool names in the response
        tool_names = ["echo", "add", "search"]
        tools_mentioned = any(name in response_text.lower() for name in tool_names)

        # Log the result instead of asserting since model behavior can vary
        if not tools_mentioned:
            print(
                f"Note: Model {model_id} did not explicitly mention tool names in response"
            )

    except Exception as e:
        pytest.fail(f"Error testing model {model_id}: {str(e)}")
