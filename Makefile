all: prepare

write_dependencies:
	@pip freeze | grep -v deep_sentence > requirements.txt

prepare:
	@pip install -r requirements.txt
	@python setup.py develop

migrate:
	@./bin/manage_db upgrade

populate_db:
	@./bin/populate_db

init_db:
	@./bin/manage_db version_control

setup_db: init_db migrate populate_db
