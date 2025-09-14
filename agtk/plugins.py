import logging
import pluggy

from agtk import hookspecs

logger = logging.getLogger(__name__)


pm = pluggy.PluginManager("agtk")
pm.add_hookspecs(hookspecs)

_loaded = False


def load_plugins():
    global _loaded
    if _loaded:
        return
    _loaded = True

    logger.debug(f"Loading plugins, before: {pm.get_plugins()}")

    import importlib.metadata

    eps = importlib.metadata.entry_points()

    # Handle different Python versions and entry_points structures
    if hasattr(eps, "select"):
        # Python 3.10+ style
        agtk_eps = eps.select(group="agtk")
        logger.debug(f"Found agtk entry points: {list(agtk_eps)}")
    elif isinstance(eps, dict):
        # Older style
        if "agtk" in eps:
            logger.debug(f"Found agtk entry points: {eps['agtk']}")
    else:
        # Python 3.12+ style with different structure
        for group in eps.groups:
            if group == "agtk":
                logger.debug("Found agtk entry points in group")

    pm.load_setuptools_entrypoints("agtk")

    logger.debug(f"Loaded: {pm.get_plugins()}")
