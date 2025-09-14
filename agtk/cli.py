import click
import sys
from runpy import run_module
import logging

from agtk.plugins import load_plugins, pm
from agtk import Toolkit
from agtk.types import ToolkitSuite

logger = logging.getLogger(__name__)


@click.group()
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging for agtk and plugins.",
)
def cli(debug: bool):
    """agtk (Agent Toolkit) offers common tools to build tools for LLM"""
    if debug:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)


@cli.command()
def mcp():
    """Start agtk as an MCP server with all available tools"""
    from fastmcp import FastMCP

    tool_suite = ToolkitSuite(toolkits=[])

    def register(toolkit: Toolkit):
        tool_suite.add_toolkit(toolkit)

    load_plugins()
    pm.hook.register_toolkit(register=register)

    if not tool_suite.get_tools():
        logger.warning(
            "No toolkits / tools found, install tools by `agtk install <tool_name>`"
        )

    mcp = FastMCP()
    tool_suite.register_mcp(mcp)
    mcp.run()


@cli.command()
@click.argument("packages", nargs=-1, required=False)
@click.option(
    "-U", "--upgrade", is_flag=True, help="Upgrade packages to latest version"
)
@click.option(
    "-e",
    "--editable",
    is_flag=True,
    help="Install packages in editable mode",
)
@click.option(
    "--force-reinstall",
    is_flag=True,
    help="Reinstall all packages even if they are already up-to-date",
)
@click.option(
    "--no-cache-dir",
    is_flag=True,
    help="Disable the cache",
)
@click.pass_context
def install(ctx, packages, upgrade, editable, force_reinstall, no_cache_dir):
    """Install packages into the same environment as agtk"""
    if ctx.parent and ctx.parent.params.get("debug"):
        logger.debug("Install command called with debug mode.")
    if not packages:
        click.echo("Error: No packages specified")
        sys.exit(1)

    args = ["pip", "install"]
    if upgrade:
        args.append("--upgrade")
    if editable:
        args.append("--editable")
    if force_reinstall:
        args.append("--force-reinstall")
    if no_cache_dir:
        args.append("--no-cache-dir")
    args.extend(packages)

    # Debug output
    click.echo(f"Running: {' '.join(args)}")

    sys.argv = args
    run_module("pip", run_name="__main__")


@cli.command()
@click.option(
    "-d",
    "--detail",
    is_flag=True,
    help="Show detail of the tools",
)
@click.pass_context
def list(ctx, detail: bool):
    """List all available tools"""
    if ctx.parent and ctx.parent.params.get("debug"):
        logger.debug("List command called with debug mode.")
    load_plugins()

    suite = ToolkitSuite(toolkits=[])

    def register(toolkit: Toolkit):
        suite.add_toolkit(toolkit)

    pm.hook.register_toolkit(register=register)

    if toolkits := suite.get_toolkits():
        for toolkit in toolkits:
            print(f"Toolkit: {toolkit.name}")
            for tool in toolkit.get_tools():
                print(f"  Tool: {tool.name}")
                if detail:
                    print(f"    Desc: {tool.description}")
                    print(f"    Params: {tool.parameters}")
    else:
        print(
            "No toolkits installed. You can install toolkits by `agtk install <tool_name>`"
        )
