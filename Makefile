.PHONY: install lint format typecheck test coverage examples check app

install:        ## installation en mode développement
	pip install -e ".[dev]"

lint:           ## analyse statique (ruff)
	ruff check .
	ruff format --check .

format:         ## formatage automatique
	ruff check . --fix
	ruff format .

typecheck:      ## typage statique (mypy)
	mypy mcfin

test:           ## suite de tests
	pytest -q

coverage:       ## tests + couverture de code (branches)
	pytest -q --cov=mcfin --cov-report=term --cov-fail-under=90

examples:       ## exécute tous les exemples (régénère docs/figures)
	cd examples && for f in [0-9]*.py; do echo "== $$f"; python $$f || exit 1; done

check: lint typecheck coverage   ## tout ce que vérifie la CI

app:            ## interface web locale (http://localhost:8000/app/)
	python app/make_manifest.py
	python app/serve.py
