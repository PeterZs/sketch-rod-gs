# Makefileの存在するディレクトリ
MAKEFILE_DIR:=$(dir $(abspath $(lastword $(MAKEFILE_LIST))))
# 一つ上のディレクトリ
PARENT_DIR := $(shell dirname ${MAKEFILE_DIR})

build:
	bash ${PARENT_DIR}/docker_environments/gaussian_splatting_train/build_script.sh

up:
	bash ${PARENT_DIR}/docker-environments/gaussian_splatting_train/up_script.sh $(DISPLAY) gs-string-env-publish

in:
	docker start gs-string-env-publish
	docker exec -it gs-string-env-publish bash

allow-gui-host:
	xhost + local:

setup:
	conda env create --file environment.yml
	conda activate splatting_viewer

update-cuda-rasterizer: 
	pip uninstall diff_gaussian_rasterization_for_sketchrodgs -y
	pip install submodules/diff-gaussian-rasterization-for-sketchrodgs/ --no-cache-dir

init-dev-venv:
	python3 -m venv gs-string-venv

activate-venv:
	source gs-string-venv/bin/activate
