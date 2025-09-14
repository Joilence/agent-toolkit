# `agtk` Toolkit Plugin

The `agtk` Toolkit plugin allows for extending the toolkit collection through plugins.

## Install a plugin

```bash
agtk install <plugin_name>
```

## Develop a plugin

```python
import agtk


@agtk.hookimpl
def register_toolkit(register):
    register(MyToolkit())

class MyToolkit(agtk.Toolkit):

    def _echo_helper(self, content: str) -> str:
        return f"echo: {content}"

    @agtk.tool_def(
        name="echo",
        description="Echo a message",
    )
    def echo(self, content: str) -> str:
        return self._echo_helper(content)

    @agtk.tool_def(
        name="echo_twice",
        description="Echo a message twice",
    )
    def echo_twice(self, content: str) -> str:
        return self._echo_helper(content) + self._echo_helper(content)
```

You can install the plugin with:

```bash
agtk install -e .
# or
agtk install -e <path_to_plugin_dir>

# check if the plugin is installed
agtk plugins
```
