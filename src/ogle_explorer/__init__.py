"""OGLE explorer: interactive period search for OGLE light curves (NKUA MSc Observational Astrophysics).

Commands installed by pip:
    ogle-explorer        Qt interface (PySide6 + pyqtgraph)
    ogle-explorer-mpl    matplotlib-widgets interface

Library use:
    from ogle_explorer import core, tools
    t, I, sigma = core.load("rxj0520.dat")
"""
try:
    from importlib.metadata import version as _version
    __version__ = _version("ogle-explorer")
except Exception:                                   # running from a source checkout without installing
    __version__ = "0.1.0"
