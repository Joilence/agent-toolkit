# Create Toolkits with Best Practices

## Tool, Toolkit and ToolkitSuite

## Automatic Schema Generation from Typings

The schema of this tool will be generated automatically if define with `@agtk.tool_def()`:

```python
from agtk import Toolkit
from pydantic import Field
from typing import Annotated

@agtk.tool_def()
def my_tool(
    param1: Annotated[int, Field(description="A required integer parameter")],
    param2: Annotated[str, Field(description="A required string parameter")],
):
    """A tool with required parameters"""
    return f"Executed with {param1} and {param2}"
```

However, following data types will have validation issues:

- `Optional[int]`
- `Optional[float]`
- `Optional[list[float]]`
- `Optional[bool]`
- `Optional[list[int]]`
- `Optional[list[str]]` (`Optional[str]` works)
- `Optional[list[bool]]`
- BaseModel (`list[BaseModel]` works)

There are a few workarounds to fix the validation issues.

For `Optional[int]`, `Optional[float]`, `Optional[bool]`, simply give a default value to the parameter:

```python
# Instead of
@agtk.tool_def()
def my_tool(
    param1: Annotated[Optional[int], Field(description="A required integer parameter")],
): ...

# Do
@agtk.tool_def()
def my_tool(
    param1: Annotated[int, Field(description="A required integer parameter")] = 0,
): ...
```

For `Optional[list[int]]`, `Optional[list[float]]`, `Optional[list[bool]]`, similarly give a default value to the parameter:

```python
# Instead of
@agtk.tool_def()
def my_tool(
    param1: Annotated[Optional[list[int]], Field(description="A required list of integers")],
): ...

# Do
@agtk.tool_def()
def my_tool(
    param1: Annotated[list[int], Field(default_factory=list, description="A required list of integers")],
): ...
```

For `BaseModel`, use `BaseModel | str` instead and then use `json.loads` to parse the string to the `BaseModel`:

```python
# Instead of
@agtk.tool_def()
def my_tool(
    param1: Annotated[BaseModel, Field(description="A required base model parameter")],
): ...


# Do
@agtk.tool_def()
def my_tool(
param1: Annotated[BaseModel | str, Field(description="A required base model parameter")],
):
    if isinstance(param1, str):
        param1 = BaseModel.model_validate_json(param1)
    ...

```

<details>
<summary>Tested on 2025-08-31 with Claude Code</summary>

```text
> Test all structured param tools from test-fastmcp with all possible params

⏺ I'll test all structured param tools from test-fastmcp with various parameter combinations. Let me start by examining the available tools and then systematically test them.

⏺ Now I'll start testing the int_tool with different integer values:

⏺ test-fastmcp - int_tool (MCP)(params: 42)
  ⎿  42

⏺ test-fastmcp - int_tool (MCP)(params: -100)
  ⎿  -100

⏺ test-fastmcp - int_tool (MCP)(params: 0)
  ⎿  0

⏺ test-fastmcp - int_tool (MCP)(params: 999999)
  ⎿  999999

⏺ Now testing optional_int_tool with null and integer values:

⏺ test-fastmcp - optional_int_tool (MCP)(params: "null")
  ⎿  Error: Input validation error: 'null' is not valid under any of the given schemas

⏺ test-fastmcp - optional_int_tool (MCP)(params: "123")
  ⎿  Error: Input validation error: '123' is not valid under any of the given schemas

⏺ test-fastmcp - optional_int_tool (MCP)(params: "-456")
  ⎿  Error: Input validation error: '-456' is not valid under any of the given schemas

⏺ Let me check the schema again - I need to pass the actual null value, not the string "null":

⏺ test-fastmcp - optional_int_tool (MCP)(params: "123")
  ⎿  Error: Input validation error: '123' is not valid under any of the given schemas

⏺ test-fastmcp - optional_int_tool (MCP)(params: "-456")
  ⎿  Error: Input validation error: '-456' is not valid under any of the given schemas

⏺ The schema shows "anyOf": [{"type": "integer"}, {"type": "null"}], so let me try with proper null value:

⏺ test-fastmcp - optional_int_tool (MCP)(params: "")
  ⎿  Error: Input validation error: '' is not valid under any of the given schemas

⏺ The optional_int_tool seems to have validation issues. Let me move on to testing default_int_tool:

⏺ Testing default_int_tool with and without parameters:
  ⎿  1

⏺ test-fastmcp - default_int_tool (MCP)(params: 777)
  ⎿  777

⏺ test-fastmcp - default_int_tool (MCP)(params: -999)
  ⎿  -999

⏺ Testing list_ints_tool with various integer arrays:

⏺ test-fastmcp - list_ints_tool (MCP)(params: [1,2,3,4,5])
  ⎿  [
       1,
       2,
     … +4 lines (ctrl+r to expand)

⏺ test-fastmcp - list_ints_tool (MCP)(params: [])


⏺ test-fastmcp - list_ints_tool (MCP)(params: [-100,0,100,999])
  ⎿  [
       -100,
       0,
     … +3 lines (ctrl+r to expand)

⏺ test-fastmcp - list_ints_tool (MCP)(params: [42])
  ⎿  [
       42
     ]

⏺ Continue with the remaining tests. Let me batch the next several tests to move efficiently through all the tools:

⏺ Testing optional_list_ints_tool and continuing with the remaining tests:

⏺ test-fastmcp - optional_list_ints_tool (MCP)(params: "[10, 20, 30]")
  ⎿  Error: Input validation error: '[10, 20, 30]' is not valid under any of the given schemas


⏺ test-fastmcp - default_list_ints_tool (MCP)(params: [99,88,77])
  ⎿  [
       99,
       88,
     … +2 lines (ctrl+r to expand)

⏺ test-fastmcp - float_tool (MCP)(params: 3.14159)
  ⎿  3.14159

⏺ test-fastmcp - float_tool (MCP)(params: -2.718)
  ⎿  -2.718

⏺ test-fastmcp - float_tool (MCP)(params: 0)
  ⎿  0

⏺ Continuing with more tests:

⏺ test-fastmcp - optional_float_tool (MCP)(params: "1.23")
  ⎿  Error: Input validation error: '1.23' is not valid under any of the given schemas
  ⎿  1

⏺ test-fastmcp - default_float_tool (MCP)(params: 5.67)
  ⎿  5.67

⏺ test-fastmcp - list_floats_tool (MCP)(params: [1.1,2.2,3.3])
  ⎿  [
       1.1,
       2.2,
     … +2 lines (ctrl+r to expand)

⏺ test-fastmcp - optional_list_floats_tool (MCP)(params: "[4.4, 5.5]")
  ⎿  Error: Input validation error: '[4.4, 5.5]' is not valid under any of the given schemas


⏺ Continuing with string tools:

⏺ test-fastmcp - str_tool (MCP)(params: "hello world")
  ⎿  hello world

⏺ test-fastmcp - str_tool (MCP)(params: "")
  ⎿

⏺ test-fastmcp - str_tool (MCP)(params: "special chars: !@#$%^&*()")
  ⎿  special chars: !@#$%^&*()

⏺ test-fastmcp - optional_str_tool (MCP)(params: "test string")
  ⎿  test string
  ⎿  default_str_tool

⏺ test-fastmcp - default_str_tool (MCP)(params: "custom string")
  ⎿  custom string

⏺ test-fastmcp - list_strs_tool (MCP)(params: ["apple","banana","cherry"])
  ⎿  [
       "apple",
       "banana",
     … +2 lines (ctrl+r to expand)

⏺ test-fastmcp - list_strs_tool (MCP)(params: [])


⏺ test-fastmcp - optional_list_strs_tool (MCP)(params: "[\"one\", \"two\"]")
  ⎿  Error: Input validation error: '["one", "two"]' is not valid under any of the given schemas


⏺ test-fastmcp - default_list_strs_tool (MCP)(params: ["custom","list"])
  ⎿  [
       "custom",
       "list"
     ]

⏺ test-fastmcp - bool_tool (MCP)(params: true)
  ⎿  true

⏺ Continuing with boolean tools:

⏺ test-fastmcp - bool_tool (MCP)(params: false)
  ⎿  false

⏺ test-fastmcp - optional_bool_tool (MCP)(params: "true")
  ⎿  Error: Input validation error: 'true' is not valid under any of the given schemas
  ⎿  true

⏺ test-fastmcp - default_bool_tool (MCP)(params: false)
  ⎿  false

⏺ test-fastmcp - list_bools_tool (MCP)(params: [true,false,true])
  ⎿  [
       true,
       false,
     … +2 lines (ctrl+r to expand)

⏺ test-fastmcp - list_bools_tool (MCP)(params: [])


⏺ test-fastmcp - optional_list_bools_tool (MCP)(params: "[false, true]")
  ⎿  Error: Input validation error: '[false, true]' is not valid under any of the given schemas


⏺ test-fastmcp - default_list_bools_tool (MCP)(params: [true,true,false])
  ⎿  [
       true,
       true,
     … +2 lines (ctrl+r to expand)

⏺ test-fastmcp - mix_types_tool (MCP)(str_param: "hello", int_param: 42, bool_param: true)
  ⎿  {
       "str_param": "hello",
       "int_param": 42,
     … +2 lines (ctrl+r to expand)

⏺ test-fastmcp - mix_types_tool (MCP)(str_param: "test", int_param: -100, bool_param: false)
  ⎿  {
       "str_param": "test",
       "int_param": -100,
     … +2 lines (ctrl+r to expand)
  ⎿  {
       "optional_str_param": null,
       "optional_int_param": null,
     … +2 lines (ctrl+r to expand)

⏺ Finishing the remaining tests:

⏺ test-fastmcp - optional_mix_types_tool (MCP)(optional_str_param: "test", optional_int_param: "123", optional_bool_param: "true")
  ⎿  Error: Input validation error: '123' is not valid under any of the given schemas
  ⎿  {
       "default_str_param": "default_str_param",
       "default_int_param": 1,
     … +2 lines (ctrl+r to expand)

⏺ test-fastmcp - default_mix_types_tool (MCP)(default_str_param: "custom", default_int_param: 999, default_bool_param: false)
  ⎿  {
       "default_str_param": "custom",
       "default_int_param": 999,
     … +2 lines (ctrl+r to expand)

⏺ test-fastmcp - structured_params_tool (MCP)(params: "{\n  \"int_param\": 42,\n  \"float_param\": 3.14,\n  \"bool_param\": true,\n  \"str_param\": \"hello world\",\n  \"list_strs_param\": [\"a\", \"b\", \"c\"],\n
                                             \"list_ints_param\": [1, 2, 3],\n  \"list_bools_param\": [true, false, true],\n  \"list_floats_param\": [1.1, 2.2, 3.3]\n}")
  ⎿  Error: Input validation error: '{\n  "int_param": 42,\n  "float_param": 3.14,\n  "bool_param": true,\n  "str_param": "hello world",\n  "list_strs_param": ["a", "b", "c"],\n  "list_ints_param": [1, 2, 3],\n
     "list_bools_param": [true, false, true],\n  "list_floats_param": [1.1, 2.2, 3.3]\n}' is not of type 'object'

⏺ Let me fix the structured params test with proper JSON object format:

⏺ test-fastmcp - structured_params_tool (MCP)(params: "{\n  \"int_param\": 42,\n  \"float_param\": 3.14,\n  \"bool_param\": true,\n  \"str_param\": \"hello world\",\n  \"list_strs_param\": [\"a\", \"b\", \"c\"],\n
                                             \"list_ints_param\": [1, 2, 3],\n  \"list_bools_param\": [true, false, true],\n  \"list_floats_param\": [1.1, 2.2, 3.3],\n  \"default_str_param\": \"default test\",\n
                                             \"default_int_param\": 100,\n  \"default_bool_param\": false,\n  \"default_list_strs_param\": [\"x\", \"y\"],\n  \"default_list_ints_param\": [10, 20],\n
                                             \"default_list_bools_param\": [false, true],\n  \"default_list_floats_param\": [5.5, 6.6]\n}")
  ⎿  Error: Input validation error: '{\n  "int_param": 42,\n  "float_param": 3.14,\n  "bool_param": true,\n  "str_param": "hello world",\n  "list_strs_param": ["a", "b", "c"],\n  "list_ints_param": [1, 2, 3],\n
     "list_bools_param": [true, false, true],\n  "list_floats_param": [1.1, 2.2, 3.3],\n  "default_str_param": "default test",\n  "default_int_param": 100,\n  "default_bool_param": false,\n  "default_list_strs_param": ["x",
     "y"],\n  "default_list_ints_param": [10, 20],\n  "default_list_bools_param": [false, true],\n  "default_list_floats_param": [5.5, 6.6]\n}' is not of type 'object'

⏺ test-fastmcp - list_structured_params_tool (MCP)(params:
                                                  [{"int_param":1,"float_param":1.1,"bool_param":true,"str_param":"first","list_strs_param":["a"],"list_ints_param":[1],"list_bools_param":[true],"list_floats_param":[1]},{"int_pa
                                                  ram":2,"float_param":2.2,"bool_param":false,"str_param":"second","list_strs_param":["b","c"],"list_ints_param":[2,3],"list_bools_param":[false,true],"list_floats_param":[2,3]}])
  ⎿  [
       {
         "int_param": 1,
     … +55 lines (ctrl+r to expand)

⏺ All structured param tools from test-fastmcp have been tested with comprehensive parameter combinations:

  Test Results Summary:

  Working Tools:
  - int_tool - Returns integer values correctly
  - default_int_tool - Uses default value (1) when no params, accepts custom values
  - list_ints_tool - Handles integer arrays including empty arrays
  - default_list_ints_tool - Uses empty array default, accepts custom arrays
  - float_tool - Handles float values with proper formatting
  - default_float_tool - Uses default value (1.0), accepts custom values
  - list_floats_tool - Handles float arrays correctly
  - default_list_floats_tool - Uses empty array default, accepts custom arrays
  - str_tool - Handles strings including empty strings and special characters
  - optional_str_tool - Accepts string values
  - default_str_tool - Uses default value ("default_str_tool"), accepts custom values
  - list_strs_tool - Handles string arrays including empty arrays
  - default_list_strs_tool - Uses empty array default, accepts custom arrays
  - bool_tool - Handles true/false values
  - default_bool_tool - Uses default value (true), accepts custom values
  - list_bools_tool - Handles boolean arrays including empty arrays
  - default_list_bools_tool - Uses empty array default, accepts custom arrays
  - mix_types_tool - Returns JSON object with all provided parameters
  - optional_mix_types_tool - Returns JSON with null values for optional parameters
  - default_mix_types_tool - Uses default values when no params, accepts custom values
  - list_structured_params_tool - Handles arrays of complex parameter objects

  Tools with Validation Issues:
  - optional_int_tool - Schema validation errors with both null and integer values
  - optional_list_ints_tool - Schema validation errors with array values
  - optional_float_tool - Schema validation errors with float values
  - optional_list_floats_tool - Schema validation errors with array values
  - optional_list_strs_tool - Schema validation errors with string arrays
  - optional_bool_tool - Schema validation errors with boolean values
  - optional_list_bools_tool - Schema validation errors with boolean arrays
  - structured_params_tool - Schema validation errors with object parameters (expects actual object, not JSON string)

  The optional parameter tools appear to have schema validation issues that prevent proper null/value handling.
```

</details>

## Relevant Links

### Primary GitHub Issues (Claude Code Repository)

- [[BUG] MCP FastMCP Pydantic model parameters serialized as JSON strings causing validation errors · Issue #3084 · anthropics/claude-code](https://github.com/anthropics/claude-code/issues/3084)
- [[CRITICAL] MCP Parameter Serialization Bug in Both Claude Code and Claude Desktop · Issue #4192 · anthropics/claude-code](https://github.com/anthropics/claude-code/issues/4192)
- [[BUG] Claude Code incorrectly serializes JSON objects as strings when calling MCP tools · Issue #5504 · anthropics/claude-code](https://github.com/anthropics/claude-code/issues/5504)
- [[BUG] Claude Code MCP Parameter Serialization Bug · Issue #3966 · anthropics/claude-code](https://github.com/anthropics/claude-code/issues/3966)
- [[BUG] Claude Code no longer passing parameters to MCP · Issue #2089 · anthropics/claude-code](https://github.com/anthropics/claude-code/issues/2089)

### FastMCP Related Issues

- [Expose Tool Input Parameter Pydantic Validation Error to the MCP Client? · Issue #1316 · jlowin/fastmcp](https://github.com/jlowin/fastmcp/issues/1316)

### Community Discussions

- [Issue with MCP Server and Pydantic Model object as tool parameter in Cursor](https://forum.cursor.com/t/issue-with-mcp-server-and-pydantic-model-object-as-tool-parameter-in-cursor/77110)
- [MCP Parameter Serialization Bug in Both Claude Code and Claude Desktop | Hacker News](https://news.ycombinator.com/item?id=44659321)

### Official Documentation & Specifications

- [Model Context Protocol Specification](https://modelcontextprotocol.io/specification)
- [MCP Tools Specification](https://spec.modelcontextprotocol.io/specification/2024-11-05/server/tools/)
- [MCP Schema JSON](https://github.com/modelcontextprotocol/specification/blob/main/schema/2024-11-05/schema.json)

### Related Projects & SDKs

- [Official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [FastMCP Repository](https://github.com/jlowin/fastmcp)
- [FastMCP Tools Documentation](https://gofastmcp.com/servers/tools)

### Anthropic Official Resources

- [Connect Claude Code to tools via MCP](https://docs.anthropic.com/en/docs/claude-code/mcp)
- [Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol)
- [Claude Code Release Notes](https://docs.anthropic.com/en/release-notes/claude-code)

### Technical Implementation Details

- [How is JSON-RPC used in the Model Context Protocol?](https://milvus.io/ai-quick-reference/how-is-jsonrpc-used-in-the-model-context-protocol)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)
- [Why Model Context Protocol uses JSON-RPC](https://medium.com/@dan.avila7/why-model-context-protocol-uses-json-rpc-64d466112338)

### AWS Related

- [AWS MCP Lambda Handler: Add support for pydantic data classes](https://github.com/awslabs/mcp/issues/684)

## Appendix: Fix Attempt

Attempted to fix the issue by overriding `fastmcp.tools.FunctionTool.run` to use `json.loads` to parse the string to the `BaseModel` if the parameter is a string, but validation issue persists before reaching this point. We must also patch tool parameter schema passed to Claude Code where `BaseModel` exists as a parameter to add `anyOf` to accept string or object, but this is essentially the same as the workaround mentioned above, so not implemented.
