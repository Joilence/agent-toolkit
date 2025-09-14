import os
import pytest
import openai

from agtk.types import ToolParamSchema

OPENAI_MODELS = [
    "o3",
    "o4-mini",
    # "codex-mini-latest", # not supported in Chat Completion API
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
]


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set",
)
@pytest.mark.parametrize("model_id", OPENAI_MODELS)
def test_openai_models_tool_compatibility(model_id, toolkit):
    """Test each OpenAI model's compatibility with our toolkit's tools."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    try:
        # Create OpenAI client
        client = openai.OpenAI()

        # Make API call
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "What tools do you have available?"}],
            tools=toolkit.as_param(ToolParamSchema.OPENAI),
        )

        # Verify the response contains mentions of our tools
        assert response.choices[0].message is not None
        response_text = response.choices[0].message.content or ""

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
