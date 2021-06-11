# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('../src'))


# -- Project information -----------------------------------------------------

project = 'repast4py'
copyright = '2021, Nick Collier, Jonathan Ozik, Eric Tatara, Sara Rimer'
author = 'Nick Collier, Jonathan Ozik, Eric Tatara, Sara Rimer'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']

# napolean settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

autodoc_inherit_docstrings = True
autoclass_content = 'both'
autodoc_typehints = 'description'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'nature'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


def setup(app):
    # repast4py.space imports various objects from the _space extension
    # module. In order to document these as being in the space module
    # we set space's __all__ here, so that when autodoc does 'from space import *'
    # to determine what to pydoc, the objects from space are returned.
    import repast4py.space
    repast4py.space.__all__ = ['DiscretePoint', 'ContinuousPoint', 'GridStickyBorders',
                               'GridPeriodicBorders', 'CartesianTopology', 
                               'BorderType', 'OccupancyType', 'SharedGrid', 'SharedCSpace',
                               'Grid', 'ContinuousSpace']
    import repast4py.core
    repast4py.core.__all__ = ['Agent', 'GhostAgent', 'GhostedAgent', 'AgentManager',
                              'SharedProjection', 'BoundedProjection']
