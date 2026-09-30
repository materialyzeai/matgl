"""
Pyinvoke tasks.py file for automating releases and admin stuff.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
from pprint import pprint

import requests
from invoke import task
from monty.os import cd


@task
def make_tutorials(ctx):
    ctx.run("rm -rf docs/tutorials")
    ctx.run("jupyter nbconvert examples/*.ipynb --to=markdown --output-dir=docs/tutorials")
    for fn in glob.glob("docs/tutorials/*/*.png"):
        ctx.run(f'mv "{fn}" docs/assets')

    for fn in os.listdir("docs/tutorials"):
        lines = ["---", "layout: default", "title: " + fn, "nav_exclude: true", "---", ""]
        path = f"docs/tutorials/{fn}"
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif fn.endswith(".md"):
            with open(path) as file:
                for line in file:
                    line = line.rstrip()
                    if line.startswith("![png]"):
                        t1, t2 = line.split("(")
                        t2, t3 = t2.split("/")
                        lines.append(t1 + "(assets/" + t3)
                    else:
                        lines.append(line)
            with open(path, "w") as file:
                file.write("\n".join(lines))


@task
def make_docs(ctx):
    """
    This new version requires markdown builder.

        pip install sphinx-markdown-builder

    Adding the following to conf.py

        extensions = [
            'sphinx_markdown_builder'
        ]

    Build markdown files with sphinx-build command

        sphinx-build -M markdown ./ build
    """
    make_tutorials(ctx)

    with cd("docs"):
        ctx.run("rm matgl.*.rst", warn=True)
        ctx.run("sphinx-apidoc -P -M -d 6 -o . -f ../src/matgl")
        # ctx.run("rm matgl*.html", warn=True)
        # ctx.run("sphinx-build -b html . ../docs")  # HTML building.
        ctx.run("sphinx-build -M markdown . .")
        ctx.run("rm *.rst", warn=True)
        ctx.run("cp markdown/matgl*.md .")
        for fn in glob.glob("matgl*.md"):
            with open(fn) as f:
                lines = [line.rstrip() for line in f if "Submodules" not in line]
            if fn == "matgl.md":
                preamble = ["---", "layout: default", "title: API Documentation", "nav_order: 5", "---", ""]
            else:
                preamble = ["---", "layout: default", "title: " + fn, "nav_exclude: true", "---", ""]
            with open(fn, "w") as f:
                f.write("\n".join(preamble + lines))

        ctx.run("rm -r markdown", warn=True)
        ctx.run("cp ../*.md .")
        ctx.run("mv README.md index.md")
        ctx.run("rm -rf *.orig doctrees", warn=True)

        with open("index.md") as f:
            contents = f.read()
        with open("index.md", "w") as f:
            contents = re.sub(
                r"\n## Official Documentation[^#]*",
                "{: .no_toc }\n\n## Table of contents\n{: .no_toc .text-delta }\n* TOC\n{:toc}\n\n",
                contents,
            )
            contents = "---\nlayout: default\ntitle: Home\nnav_order: 1\n---\n\n" + contents

            f.write(contents)


@task
def publish(ctx):
    ctx.run("rm dist/*.*", warn=True)
    ctx.run("python setup.py sdist bdist_wheel")
    ctx.run("twine upload dist/*")


@task
def release(ctx, version):
    desc = get_changelog(version)
    payload = {
        "tag_name": "v" + version,
        "target_commitish": "main",
        "name": "v" + version,
        "body": desc,
        "draft": False,
        "prerelease": False,
    }
    response = requests.post(
        "https://api.github.com/repos/materialyzeai/matgl/releases",
        data=json.dumps(payload),
        headers={"Authorization": "token " + os.environ["GITHUB_RELEASES_TOKEN"]},
    )
    pprint(response.json())


def get_changelog(version):
    with open("docs/changes.md") as f:
        contents = f.read()
        m = re.search(rf"^## {re.escape(version)}\n(.*?)(?=^## |\Z)", contents, re.DOTALL | re.MULTILINE)
        changes = m.group(1).strip()
        return changes


@task
def view_docs(ctx):
    with cd("docs"):
        ctx.run("bundle exec jekyll serve")
