all: prepare

write_dependencies:
	@pip freeze | grep -v deep_sentence > requirements.txt

prepare:
	@pip install -r requirements.txt
	@python setup.py develop
