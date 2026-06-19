# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A personal, **interview-driven + project-based** learning repo for becoming an AI/LLM engineer. The content is **primarily in Chinese** — match that language in new notes, docstrings, and commit messages. Every learning unit follows the loop in `docs/学习方法论.md`: *real interview question → can't answer → build a minimal project to learn the theory → commit code + notes + your own answer here*.

**第一原则（最高优先级）：保证项目优先，在项目中把知识点学清楚。** Every knowledge point must be learned by building a runnable project — never by adding theory/notes alone. When this principle conflicts with the repo's existing "pure-numpy, zero-dep" style, **project-first wins**: if a concept (e.g. SFT training) can only be learned clearly by actually running it, pull in real frameworks (PyTorch). Keep the numpy "原理版" and add a `*_real.py` alongside (see `04-RAG/mini_rag`). Don't get stuck in gap-analysis or framework comparisons — move to a project fast.

There is **no application, package, or test suite** — outputs are individual runnable Python scripts plus Markdown knowledge files. Don't introduce a build system, lint config, or test framework unless asked.

## Two parallel tracks (the big picture)

1. **Theory track → `knowledge/`** — a sourced knowledge base, one entry per module. The structure of every knowledge file is fixed: `【参考资料来源】` → 正文 → `【面试考点】` → `【易错/陷阱】` → `【开源覆盖缺口】`. Every non-trivial claim is cited inline as `[来源 §x]` so it is verifiable. Do not paraphrase claims without keeping a source.
   - `knowledge/RAG/` (📕) is the most complete — 13 chapters transcribed from a physical book (严灿平《RAG应用开发与优化》).
   - The other five modules (`Transformer/ LLM基础/ Prompt工程/ Agent/ 微调与部署/`, each a single `系统知识.md`, 📄) were rebuilt from primary papers (arXiv) + official docs — citations point to arXiv numbers and doc sections.

2. **Practice track → `00-数学基础/` … `06-微调与部署/` + `projects/`** — numbered lesson scripts that make the theory run. Each module folder typically holds `笔记.md` (ties the module's scripts into one mental map), optional `面试题.md`, and numbered `.py` lessons. `projects/` holds cross-module capstones (e.g. `智能知识助手/` = RAG + ReAct + Prompt + memory).

## Commands

Environment is managed with **uv** (Python ≥ 3.12). `pyproject.toml` declares only `numpy` and `matplotlib`; everything else is optional and intentionally **not** in deps.

```bash
uv sync                                                  # install declared deps (numpy, matplotlib)
uv run python "04-RAG/mini_rag/main.py"                  # run a script when cwd == repo root
uv run --directory /Users/sunhuanyu/ai-llm-learning python "01-Transformer/02-多头注意力.py"  # run from anywhere
```

`--directory` is needed whenever the working directory is not the repo root, because uv must locate `pyproject.toml`. There is **no test runner and no linter** — "verification" means running the relevant script(s) and checking their printed output / generated images. There is no single-test concept.

## Code conventions (follow these exactly — they're load-bearing)

- **Self-contained docstring header on every `.py`.** Each script opens with a multi-section docstring: what it teaches → references (paper title + arXiv number + URLs) → a metaphor in plain language → `【面试高频考点】` → the exact `uv run ...` command to run it. New scripts must keep this header.
- **Two-tier implementation pattern for projects.** A concept is shown twice, decoupling skeleton from components: a **pure-numpy "toy" version** (zero external deps, always runs offline — this is the canonical `uv run` target) and a **"real" version** (`*_real.py`) using chromadb / ollama / sentence-transformers. When adding a project, default to the numpy version; the real version is opt-in and documented in its folder's `README.md`. See `04-RAG/mini_rag/{main.py,main_real.py}` for the template.
- **Never auto-install optional deps.** To avoid environment conflicts, scripts do lazy imports *inside* functions (e.g. `from openai import OpenAI` only where called) and instructions tell the user to `uv pip install ...` / `ollama pull ...` themselves. Do not add `openai`, `chromadb`, `ollama`, etc. to `pyproject.toml` unless asked.
- **Chinese figures.** matplotlib scripts set macOS CJK fonts at the top — copy this verbatim:
  ```python
  plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
  plt.rcParams['axes.unicode_minus'] = False
  ```
  Generated figures live in each module's `img/` folder and are referenced from the `笔记.md`.
- **Secrets.** API keys go in a `.env` at the repo root (gitignored). Scripts read via `os.environ`; never hardcode keys.

## Where things go (anti-confusion)

- A new **theory topic** → the matching file under `knowledge/<module>/` (keep the fixed section structure + `[来源 §x]` citations; if a module only has `系统知识.md`, extend it rather than spawning files — RAG is the only chapter-per-file module).
- A new **lesson/practice script** → the matching numbered `0X-<module>/` folder; then update that folder's `笔记.md` to weave the new script into the module's mental map.
- A new **comprehensive multi-module project** → `projects/`, with its own `README.md` documenting the three "swap points" for upgrading from the numpy mock to a real model (LLM step / embedding / vector store) — follow `projects/智能知识助手/` as the template.

## Provenance & sourcing rules

The repo's stated value proposition is *reliability through traceable sources* (see `knowledge/README.md`). When writing or editing knowledge content:
- Keep arXiv numbers, paper titles, and section references accurate. The five 📄 modules were built by actually fetching the papers/docs (arXiv via ar5iv, official docs) — for new claims, prefer citing a fetched primary source over memory.
- If you are unsure whether a cited fact is correct, flag it rather than inventing a citation.
