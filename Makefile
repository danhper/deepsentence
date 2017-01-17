NPM := $(shell which yarn || which npm)
FOREMAN ?= $(shell which foreman)
PYTHON ?= python
PIP ?= pip

all: prepare

setup: python_dependencies npm_dependencies

write_dependencies:
	@$(PIP) freeze | grep -v deep_sentence > requirements.txt

npm_dependencies:
	@cd deep_sentence/webapp && $(NPM) install

python_dependencies:
	@$(PIP) install -r requirements.txt

compile_web:
	@cd deep_sentence/webapp && ./node_modules/.bin/webpack -p

webapp_setup: npm_dependencies compile_web
	@$(PYTHON) -c "import nltk; nltk.download('punkt')"


prepare: python_dependencies
	@$(PYTHON) setup.py develop

migrate:
	@./bin/manage_db upgrade

populate_db:
	@./bin/populate_db

init_db:
	@./bin/manage_db version_control

setup_db: init_db migrate populate_db

download_models:
	@./bin/download_models

dev_webapp:
	$(FOREMAN) start -f deep_sentence/webapp/Procfile

debug_webapp:
	env FLASK_APP=deep_sentence.webapp FLASK_DEBUG=1 flask run

webpack_watch:
	cd deep_sentence/webapp && ./node_modules/.bin/webpack --watch
