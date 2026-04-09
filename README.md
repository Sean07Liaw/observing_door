# lab-door-ai

Privacy-first lab/office state observation MVP.

## Strucure
obsering_door/
  app/
    __init__.py
    config.py
    db.py
    models.py
    main.py
  scripts/
    init_db.py
  data/
    images/
    clips/
  pyproject.toml
  .env.example
  README.md

## Step A
Current scope:
- project skeleton
- config
- sqlite database
- core models
- db initialization

## Setup

1. Create virtual environment
2. Install requirements
3. Copy `.env.example` to `.env`
4. Initialize database
5. Run API server

## Commands

```bash
python -m venv .venv
.venv\\Scripts\\activate