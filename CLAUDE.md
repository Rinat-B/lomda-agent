# CLAUDE.md

## Language
- Talk to me in Hebrew: explanations, questions, plans and reports.
- Code is English only: code, comments, identifiers, file names, commit messages, docs, README.
- Everything the end user sees is Hebrew: CLI messages, prompts, errors, help text, and the lesson content.
- Hebrew user-facing strings never appear inside .py files. Keep them all in agent/assets/messages_he.json and load them by key (e.g. msg("gate.topics.ask")).
- Prompts to the model are written in English and tell the model to answer in Hebrew.
- Technical terms (commit, venv, API key) may stay in English inside Hebrew explanations.
