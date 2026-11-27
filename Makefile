export PYTHONPATH := src

.PHONY: test smoke run lint
test:
	python -m pytest -q
smoke:
	python -m newsclust.run --subsample 3000 --seeds 1 --out /tmp/news-smoke
run:
	python -m newsclust.run
lint:
	ruff check .
