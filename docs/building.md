# Building the documentation

The documentation follows TreeIG's Sphinx, MyST Markdown, and PyData Sphinx
Theme conventions. NumPy-style API docstrings use Napoleon; mathematical
notation uses MathJax. Navigation keeps the complete compact guide visible.

From the repository root, using Python 3.11 or newer:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs docs/_build/html
python scripts/check_docs_discovery.py
```

Open `docs/_build/html/index.html` in a browser. Documentation dependencies are
optional and do not change CBaseline's runtime requirements. The documentation
configuration reads the version from `pyproject.toml` using Python 3.11's
`tomllib`. Generated HTML is ignored by Git.

## Examples

To run the README and integration examples, also install:

```bash
python -m pip install treeig scikit-learn shap
```

The examples use generated data and fixed seeds. They need no external dataset.
Classification snippets explain how to adapt a fitted model; they require
its actual logits, class order, and chosen reference population.

## Continuous integration and publishing

The Documentation workflow builds HTML with warnings treated as errors and
uploads a downloadable artifact on pushes and pull requests. Like TreeIG,
it deploys to GitHub Pages on pushes to `main` or manual runs on `main`.
Pull requests build without deploying.

For initial publication, select **Settings → Pages → GitHub Actions** as the
repository's Pages source, then run the workflow on `main`. The expected site
address is https://ludgerhentschel.github.io/cbaseline/ after a successful
deployment. Adding these files alone does not publish the site.

Edit topic pages under `docs/`, keeping the README focused on the package,
papers, and the first examples. Keep `docs/requirements.txt` and the `docs`
extra in `pyproject.toml` synchronized.

## Discovery files

Maintain the repository-root `llms.txt` as an annotated map to the published
guides. Sphinx copies this single source to the documentation base URL through
`html_extra_path`. Use rendered API links because raw Sphinx sources contain
autodoc instructions rather than expanded signatures and docstrings.

`sphinx-sitemap` generates `sitemap.xml` with the project prefix from
`html_baseurl`, which also supplies canonical page URLs. Important pages define
concise descriptions in their `myst.html_meta` YAML front matter.

The discovery check runs after the HTML build in CI. It verifies the copied
index, local link targets, sitemap URLs, canonical links, descriptions, and
expanded API and example content. After deployment, check the public
`/cbaseline/llms.txt` and `/cbaseline/sitemap.xml` endpoints and external links.
