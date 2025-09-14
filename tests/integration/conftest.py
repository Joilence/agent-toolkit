from typing import Union, List, Optional
import agtk
import pytest
from pydantic import BaseModel

"""
Integration test toolkit, include:
- simple tool with one parameter
- simple tool with multiple parameters
- complicated tool with pydantic basemodel for parameters
"""


class SearchQueryParams(BaseModel):
    """Parameters for search query."""

    query: str
    max_results: int = 10
    case_sensitive: bool = False
    categories: Optional[List[str]] = None


class TestIntegrationToolkit(agtk.Toolkit):
    """A test toolkit for integration testing."""

    @agtk.tool_def(name="echo", description="Echo back the input")
    def echo(self, message: str) -> str:
        """Echo back the input message."""
        return f"You said: {message}"

    @agtk.tool_def(name="add", description="Add two numbers")
    def add(self, a: Union[int, float], b: Union[int, float]) -> Union[int, float]:
        """Add two numbers and return the result."""
        return a + b

    @agtk.tool_def(name="search", description="Search for items matching the query")
    def search(self, params: SearchQueryParams) -> List[str]:
        """Search for items matching the given query parameters.

        Args:
            params: The search parameters including query string, max results,
                   case sensitivity, and optional categories to filter by.

        Returns:
            A list of matching items.
        """
        # Mock data for the search
        all_items = [
            "Apple",
            "Banana",
            "Cherry",
            "Document1",
            "Document2",
            "Document3",
            "User1",
            "User2",
            "User3",
        ]

        # Filter based on query
        if params.case_sensitive:
            results = [item for item in all_items if params.query in item]
        else:
            results = [
                item for item in all_items if params.query.lower() in item.lower()
            ]

        # Filter by categories if provided
        if params.categories:
            # Mock category filtering (in real implementation, items would have categories)
            if "fruits" in params.categories:
                results = [r for r in results if r in ["Apple", "Banana", "Cherry"]]
            elif "documents" in params.categories:
                results = [
                    r for r in results if r in ["Document1", "Document2", "Document3"]
                ]
            elif "users" in params.categories:
                results = [r for r in results if r in ["User1", "User2", "User3"]]

        # Limit results
        return results[: params.max_results]

    @agtk.tool_def(
        name="calculate_stats",
        description="Calculate basic statistics for a list of numbers",
    )
    def calculate_stats(
        self,
        numbers: List[float],
        include_median: bool = True,
        precision: int = 2,
    ) -> dict:
        """Calculate basic statistics (mean, median, min, max) for a list of numbers.

        Args:
            numbers: List of numbers to calculate statistics for
            include_median: Whether to include median in the calculation
            precision: Number of decimal places to round to

        Returns:
            Dictionary containing calculated statistics
        """
        if not numbers:
            return {"error": "Cannot calculate statistics for empty list"}

        stats = {
            "count": len(numbers),
            "mean": round(sum(numbers) / len(numbers), precision),
            "min": round(min(numbers), precision),
            "max": round(max(numbers), precision),
        }

        if include_median:
            sorted_nums = sorted(numbers)
            n = len(sorted_nums)
            if n % 2 == 0:
                median = (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
            else:
                median = sorted_nums[n // 2]
            stats["median"] = round(median, precision)

        return stats


@pytest.fixture
def toolkit():
    """Fixture to provide a TestIntegrationToolkit instance."""
    return TestIntegrationToolkit()
