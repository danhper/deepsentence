all: prepare

setup: python_dependencies

write_dependencies:
	@pip freeze | grep -v deep_sentence > requirements.txt

python_dependencies:
	@pip install -r requirements.txt

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
	foreman start --procfile deep_sentence/webapp/Procfile
