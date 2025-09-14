from pluggy import HookimplMarker, HookspecMarker


hookspec = HookspecMarker("ltls")
hookimpl = HookimplMarker("ltls")


@hookspec
def register_toolkit(register):
    "Register toolkit to the suite"
