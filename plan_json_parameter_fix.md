# Implementation Plan: JSON String Parameter Conversion in ltls

## Requirements

### Problem
MCP clients serialize array and dictionary parameters as JSON strings (e.g., `'["type"]'` instead of `["type"]`), causing validation failures in ltls/FastMCP when functions expect native Python types like `List[T]` or `Dict[K, V]`.

### Goal
Implement automatic JSON string to Python type conversion in ltls to handle MCP client parameter serialization transparently.

### Non-Goal
- Modifying FastMCP core library
- Changing MCP protocol specification
- Converting string parameters that are intentionally strings (even if they contain JSON)

## Functional Spec

### Current Behavior (Broken)
```python
# MCP sends: {"numbers": "[1.0, 2.5, 3.7]"}
tool.execute({"numbers": "[1.0, 2.5, 3.7]"})
# Result: ValidationError: Input should be a valid list
```

### Expected Behavior (Fixed)
```python
# MCP sends: {"numbers": "[1.0, 2.5, 3.7]"}
tool.execute({"numbers": "[1.0, 2.5, 3.7]"})
# Result: Success - numbers parsed as [1.0, 2.5, 3.7]
```

### Conversion Rules
1. **List/Array Types**: `"[1, 2, 3]"` → `[1, 2, 3]` when expecting `List[T]`
2. **Dict/Object Types**: `'{"key": "value"}'` → `{"key": "value"}` when expecting `Dict[K, V]`
3. **String Parameters**: JSON strings stay as strings when expecting `str`
4. **Optional Types**: Handle `Optional[List[T]]` and `Optional[Dict[K, V]]`
5. **Invalid JSON**: Keep as string if JSON parsing fails

## Technical Spec

### Root Cause
- MCP clients serialize complex types as JSON strings
- ltls/FastMCP TypeAdapter expects native Python types
- No preprocessing layer exists to handle this conversion

### Solution Architecture

#### Implementation Location
**File**: `/Users/jyang/localcode/ltls-worktrees/llm-tools/ltls/types.py`

#### Key Components

1. **Preprocessing Method** (`_preprocess_json_arguments`)
```python
def _preprocess_json_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
    """Convert JSON strings to Python objects where appropriate."""
    # Inspect function signature to understand expected types
    # For each parameter:
    #   - If expecting List/Dict and value is JSON string → parse
    #   - If expecting str → keep as string
    #   - Handle Optional types correctly
```

2. **Type Detection Logic**
```python
# Handle Optional[T] or Union[T, None]
if origin is Union:
    args = get_args(expected_type)
    non_none_types = [t for t in args if t is not type(None)]
    if len(non_none_types) == 1:
        expected_type = non_none_types[0]
```

3. **Safe JSON Parsing**
```python
# Only parse if string looks like JSON array/object
if origin in (list, List):
    should_parse = value.strip().startswith('[') and value.strip().endswith(']')
elif origin in (dict, Dict):
    should_parse = value.strip().startswith('{') and value.strip().endswith('}')
```

4. **Integration Points**
- `execute()` method: Add preprocessing before validation
- `aexecute()` method: Add preprocessing before validation

### Implementation Status

✅ **Completed**:
1. Added `_preprocess_json_arguments` method to Tool class
2. Integrated preprocessing in both `execute()` and `aexecute()` methods
3. Added proper imports for type introspection (`get_args`, `get_origin`)
4. Comprehensive testing confirms all edge cases work correctly

### Test Results

All 12 test scenarios pass:
- ✅ List parameters with JSON strings
- ✅ Dict parameters with JSON strings
- ✅ Optional types with JSON strings
- ✅ String parameters containing JSON (stay as strings)
- ✅ Mixed parameter types
- ✅ Invalid JSON handling
- ✅ Empty collections
- ✅ Backward compatibility with native types

### Validation Flow

```
1. MCP Client: {"group_by": '["type"]'}
2. Tool.execute() receives string
3. _preprocess_json_arguments():
   - Detects group_by expects List[str]
   - Parses JSON string → ["type"]
4. TypeAdapter.validate_python() receives native list
5. ✅ Validation passes
```

## Why Fix in ltls Instead of Individual Toolkits

### Architectural Decision
The fix belongs in ltls because:

1. **Single Point of Fix**: One implementation helps all tools
2. **Developer Experience**: Tool developers shouldn't handle JSON parsing
3. **Proper Abstraction Layer**: ltls sits between MCP protocol and user functions
4. **Consistency**: Similar to how Pydantic handles string→int conversion
5. **Backward Compatible**: Doesn't break existing tools

### Alternative Approaches Considered

1. **Fix in FastMCP**: Would be more comprehensive but requires upstream changes
2. **Fix in Each Toolkit**: Poor DX, duplicate code, error-prone
3. **Fix in MCP Clients**: Protocol-level change, harder to coordinate
4. **Fix in ltls**: ✅ Best balance of effectiveness and maintainability

## Definition of Done

- [x] Automatic JSON string conversion for List/Dict parameters
- [x] String parameters with JSON content remain unchanged
- [x] Optional type handling works correctly
- [x] Invalid JSON gracefully handled
- [x] All existing ltls tests pass
- [x] Comprehensive test coverage for edge cases
- [x] No performance regression
- [x] Backward compatibility maintained

## Next Steps

1. **For ltls maintainers**: Consider merging this fix into ltls main branch
2. **For toolkit developers**: Remove manual JSON parsing workarounds once ltls is updated
3. **For MCP ecosystem**: Consider standardizing parameter serialization in the protocol