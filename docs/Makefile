# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= 
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Set the source directory for the python module
MODULEDIR     = ../src/aerial_anomaly_detection

# First run sphinx-apidoc to generate .rst files from the source code
apidoc:
	@echo "Running sphinx-apidoc to generate .rst files..."
	@sphinx-apidoc -o $(SOURCEDIR) $(MODULEDIR)

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help apidoc Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Default target to run apidoc first, then sphinx-build
build: apidoc
	@$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR) $(SPHINXOPTS) $(O)
