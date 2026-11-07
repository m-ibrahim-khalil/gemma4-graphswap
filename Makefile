test:
	uv run pytest -q

WHEEL ?= vendor/swegemma-0.2.11-py3-none-any.whl

# Copy only swegemma/graph/ out of the wheel (the top-level package imports heavy deps).
vendor:
	mkdir -p vendor
	@if [ ! -f "$(WHEEL)" ]; then \
	  kaggle datasets download metric/gemma-4-developer-agent-wheelhouse -f swegemma-0.2.11-py3-none-any.whl -p vendor/; \
	  if [ -f "$(WHEEL).zip" ]; then \
	    mkdir -p vendor/_dl && unzip -q -o "$(WHEEL).zip" -d vendor/_dl && mv vendor/_dl/*.whl "$(WHEEL)" && rm -rf vendor/_dl "$(WHEEL).zip"; \
	  fi; \
	fi
	rm -rf vendor/swegemma vendor/_unz
	mkdir -p vendor/swegemma vendor/_unz
	unzip -q -o "$(WHEEL)" -d vendor/_unz
	cp -R vendor/_unz/swegemma/graph vendor/swegemma/graph
	touch vendor/swegemma/__init__.py
	rm -rf vendor/_unz

.PHONY: test vendor
