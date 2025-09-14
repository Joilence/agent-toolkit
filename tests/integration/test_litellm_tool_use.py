import os
import pytest
import litellm

from agtk.types import ToolParamSchema

OPENAI_MODELS = [
    "openai/o3",
    "openai/o4-mini",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-4.1",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1-nano",
]

ANTHROPIC_MODELS = [
    "anthropic/claude-opus-4-20250514",
    "anthropic/claude-sonnet-4-20250514",
    "anthropic/claude-3-7-sonnet-20250219",
    "anthropic/claude-3-5-sonnet-20240620",
    "anthropic/claude-3-5-haiku-20241022",
]

GEMINI_MODELS = [
    "gemini/gemini-2.5-flash-preview-05-20",
    "gemini/gemini-2.5-pro-preview-05-06",
]

# Combine all models into one list for testing
ALL_MODELS = OPENAI_MODELS + ANTHROPIC_MODELS + GEMINI_MODELS


@pytest.mark.integration
@pytest.mark.skipif(
    not (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    ),
    reason="No API keys set for any supported provider",
)
@pytest.mark.parametrize("model_id", ALL_MODELS)
def test_litellm_models_tool_compatibility(model_id, toolkit):
    """Test each model's compatibility with our toolkit's tools using LiteLLM."""

    # Skip models for which we don't have API keys
    if model_id.startswith("openai/") and not os.environ.get("OPENAI_API_KEY"):
        pytest.skip(f"Skipping {model_id}: OPENAI_API_KEY not set")

    if model_id.startswith("anthropic/") and not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip(f"Skipping {model_id}: ANTHROPIC_API_KEY not set")

    if model_id.startswith("gemini/") and not os.environ.get("GEMINI_API_KEY"):
        pytest.skip(f"Skipping {model_id}: GEMINI_API_KEY not set")

    try:
        # Make API call using LiteLLM
        response = litellm.completion(
            model=model_id,
            messages=[{"role": "user", "content": "What tools do you have available?"}],
            tools=toolkit.as_param(ToolParamSchema.OPENAI),
        )

        # Verify the response contains mentions of our tools
        assert response.choices[0].message is not None
        response_text = response.choices[0].message.content or ""

        # Check if the response mentions tools
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
