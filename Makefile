NPM := $(shell type -p yarn || type -p npm)
FOREMAN ?= (shell type -p foreman)

all: prepare

setup: python_dependencies npm_dependencies

write_dependencies:
	@pip freeze | grep -v deep_sentence > requirements.txt

npm_dependencies:
	@cd deep_sentence/webapp && $(NPM) install

python_dependencies:
	@pip install -r requirements.txt

prepare_web: prepare npm_dependencies

prepare: python_dependencies
	@python setup.py develop

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
	foreman start -f deep_sentence/webapp/Procfile

debug_webapp:
	env FLASK_APP=deep_sentence.webapp FLASK_DEBUG=1 flask run

webpack_watch:
	cd deep_sentence/webapp && ./node_modules/.bin/webpack --watch
