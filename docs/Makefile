# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = python3 -msphinx
SPHINXPROJ    = phyloframe
SOURCEDIR     = .
BUILDDIR      = _build

readme.rst: ../README.md
	pandoc ../README.md --wrap=none --from markdown --to rst -s -o readme.rst

citing.rst: citing.md
	pandoc citing.md --wrap=none --from markdown --to rst -s -o citing.rst

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf _autosummary _modules _build readme.rst citing.rst

.PHONY: help Makefile clean

# mark .md targets as source files that need no action
%.md:
	true

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile readme.rst citing.rst
	cp index_stage1.rst index.rst
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	cp index_stage2.rst index.rst
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
