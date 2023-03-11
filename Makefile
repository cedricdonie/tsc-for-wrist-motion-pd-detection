# Disable built-in rules and variables
MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-builtin-variables
.SUFFIXES:
.DELETE_ON_ERROR:
# Never delete the external data
.PRECIOUS: data/external
# Only delete processed data on error
.SECONDARY:
.PHONY: clean data lint requirements reports create_environment verify_data_external_filelist
.SECONDEXPANSION: models/%.pred.npz
# Use bash for brace expansion
SHELL:=/bin/bash


#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE := default
PROJECT_NAME := mjff_levadopa_analysis_cedric_donie
PYTHON_INTERPRETER := python3.6
VENV := $(HOME)/.venvs/mjff-ldopa
FIND_ARGS := -type f -printf "%p %k KB\n"

# Cloud Settings
DOCKER_REGISTRY_NAME := DOCKER_REGISTRY_NAME_HERE
GCP_PROJECT_NAME := GCP_PROJECT_NAME_HERE
GCP_BUCKET_NAME := $(GCP_PROJECT_NAME)-aiplatform
GCP_DOCKER_URI := gcr.io/$(GCP_PROJECT_NAME)/$(DOCKER_REGISTRY_NAME)

GIT_REVISION = $(shell git log -1 --format=%h)

#################################################################################
# COMMANDS                                                                      #
#################################################################################

VERIFY_DATA_EXTERNAL_FILELIST := bash -c 'diff -u data/external_filelist.txt <(find data/external/ $(FIND_ARGS) | sort | grep -vF .tsv)'
MAKE_DEPEND := mkdir -p .deps/ && python src/models/makedepend_hyperparameters.py models/inceptiontime/hyperparameter_tuning/hyperparameters.json


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Test Python install
test: requirements
	./test/doctest.sh

## Calculate metrics of model
models/%.metrics.json: models/%.pred.npz
	python src/visualization/metrics.py $^ > $@

## Predict model
models/%.pred.npz: $$(basename $$(@D)) data/processed/sktime/$$(*F).data.pkl
	mkdir -p $(@D)
	python src/models/predict_model.py $^ $@


# Train models

## Train inceptiontime model (workaround due to float precision)
models/inceptiontime/tremor_GENEActiv_train_windowsz-21.88_overlap-10.94.model.h5: data/processed/sktime/GENEActiv_train_windowsz-21.88_overlap-10.94.data.pkl data/processed/sktime/GENEActiv_val_windowsz-21.89_overlap-10.94.data.pkl
	python -u src/models/train_model.py $< $@ tremor --classifier inceptiontime --validation-data-file $(word 2,$^) > $@.txt 2>&1

## Train inceptiontime model
models/inceptiontime/tremor_%.model.h5: data/processed/sktime/%.data.pkl data/processed/sktime/$$(subst _train_,_val_,%).data.pkl
	python -u src/models/train_model.py $< $@ tremor --classifier inceptiontime --validation-data-file $(word 2,$^) > $@.txt 2>&1

## Train inceptiontime model
models/inceptiontime/dyskinesia_%.model.h5: data/processed/sktime/%.data.pkl data/processed/sktime/$$(subst _train_,_val_,%).data.pkl
	python -u src/models/train_model.py $< $@ dyskinesia --classifier inceptiontime --validation-data-file $(word 2,$^) > $@.txt 2>&1

## Train inceptiontime model
models/inceptiontime/bradykinesia_%.model.h5: data/processed/sktime/%.data.pkl data/processed/sktime/$$(subst _train_,_val_,%).data.pkl
	python -u src/models/train_model.py $< $@ bradykinesia --classifier inceptiontime --validation-data-file $(word 2,$^) > $@.txt 2>&1

## Train inceptiontime model with reduced data
models/inceptiontime/tremor_%_sampledevery-10.model.h5: data/processed/sktime/%.data.pkl
	python -u src/models/train_model.py $< $@ tremor --classifier inceptiontime --sample-every-nth-row 10 > $@.txt 2>&1

## Train inceptiontime model with reduced data
models/inceptiontime/dyskinesia_%_sampledevery-10.model.h5: data/processed/sktime/%.data.pkl
	python -u src/models/train_model.py $< $@ dyskinesia --classifier inceptiontime --sample-every-nth-row 10 > $@.txt 2>&1

## Train inceptiontime model with reduced data
models/inceptiontime/bradykinesia_%_sampledevery-10.model.h5: data/processed/sktime/%.data.pkl
	python -u src/models/train_model.py $< $@ bradykinesia --classifier inceptiontime --sample-every-nth-row 10 > $@.txt 2>&1

## Train minirocket model
models/minirocket/tremor_%.model.pkl: data/processed/sktime/%.data.pkl
	python -u src/models/train_model.py $< $@ tremor --classifier minirocket > $@.txt 2>&1

## Train minirocket model
models/minirocket/dyskinesia_%.model.pkl: data/processed/sktime/%.data.pkl
	python -u src/models/train_model.py $< $@ dyskinesia --classifier minirocket > $@.txt 2>&1

## Train minirocket model
models/minirocket/bradykinesia_%.model.pkl: data/processed/sktime/%.data.pkl
	python -u src/models/train_model.py $< $@ bradykinesia --classifier minirocket > $@.txt 2>&1

## Train rocket model
models/rocket/tremor_%.model.pkl: data/processed/sktime/%.data.pkl
	python -u src/models/train_model.py $< $@ tremor --classifier rocket > $@.txt 2>&1

## Train rocket model
models/rocket/dyskinesia_%.model.pkl: data/processed/sktime/%.data.pkl
	python -u src/models/train_model.py $< $@ dyskinesia --classifier rocket > $@.txt 2>&1

## Train rocket model
models/rocket/bradykinesia_%.model.pkl: data/processed/sktime/%.data.pkl
	python -u src/models/train_model.py $< $@ bradykinesia --classifier rocket > $@.txt 2>&1

## Train gaussian process model
models/gp/tremor_%.model.pkl: data/processed/sktime/%.data.pkl
	mkdir -p $(@D)
	python src/models/train_model.py $< $@ tremor --classifier gp > $@.txt 2>&1

## Train gaussian process model
models/gp/dyskinesia_%.model.pkl: data/processed/sktime/%.data.pkl
	mkdir -p $(@D)
	python src/models/train_model.py $< $@ dyskinesia --classifier gp > $@.txt 2>&1

## Train gaussian process model
models/gp/bradykinesia_%.model.pkl: data/processed/sktime/%.data.pkl
	mkdir -p $(@D)
	python src/models/train_model.py $< $@ bradykinesia --classifier gp > $@.txt 2>&1

# Train final models

## Final InceptionTime model for tremor incl. hyperparameter optimization
models/inceptiontimetuned/%/tremor_GENEActiv_train-val_windowsz-24.57_overlap-12.29.model.h5: data/processed/sktime/GENEActiv_train-val_windowsz-24.57_overlap-12.29.data.pkl
	mkdir -p $(@D)
	python -u src/models/train_model.py $< $@ tremor --classifier inceptiontime --epochs 600 --random-state $(*F) --use-best-params --no-checkpointing > $@.txt 2>&1

## Final InceptionTime model for dyskinesia incl. hyperparameter optimization
models/inceptiontimetuned/%/dyskinesia_GENEActiv_train-val_windowsz-29.33_overlap-14.66.model.h5: data/processed/sktime/GENEActiv_train-val_windowsz-29.33_overlap-14.66.data.pkl
	mkdir -p $(@D)
	python -u src/models/train_model.py $< $@ dyskinesia --classifier inceptiontime --epochs 600 --random-state $(*F) --use-best-params --no-checkpointing > $@.txt 2>&1

models/inceptiontimetuned/%/bradykinesia_GENEActiv_train-val_windowsz-22.37_overlap-11.18.model.h5: data/processed/sktime/GENEActiv_train-val_windowsz-22.37_overlap-11.18.data.pkl
	mkdir -p $(@D)
	python -u src/models/train_model.py $< $@ bradykinesia --classifier inceptiontime --epochs 600 --random-state $(*F) --use-best-params --no-checkpointing > $@.txt 2>&1

## Final InceptionTime default model for tremor
models/inceptiontime/%/tremor_GENEActiv_train-val_windowsz-30.0_overlap-15.0.model.h5: data/processed/sktime/GENEActiv_train-val_windowsz-30.0_overlap-15.0.data.pkl
	mkdir -p $(@D)
	python -u src/models/train_model.py $< $@ tremor --classifier inceptiontime --epochs 600 --random-state $(*F) > $@.txt 2>&1

## Final InceptionTime default model for bradykinesia
models/inceptiontime/%/bradykinesia_GENEActiv_train-val_windowsz-30.0_overlap-15.0.model.h5: data/processed/sktime/GENEActiv_train-val_windowsz-30.0_overlap-15.0.data.pkl
	mkdir -p $(@D)
	python -u src/models/train_model.py $< $@ bradykinesia --classifier inceptiontime --epochs 600 --random-state $(*F) > $@.txt 2>&1

## Final InceptionTime default model for dyskinesia
models/inceptiontime/%/dyskinesia_GENEActiv_train-val_windowsz-30.0_overlap-15.0.model.h5: data/processed/sktime/GENEActiv_train-val_windowsz-30.0_overlap-15.0.data.pkl
	mkdir -p $(@D)
	python -u src/models/train_model.py $< $@ dyskinesia --classifier inceptiontime --epochs 600 --random-state $(*F) > $@.txt 2>&1

## Final MLP model for tremor
models/gp/%/tremor_GENEActiv_train-val_windowsz-30.0_overlap-15.0.model.pkl: data/processed/sktime/GENEActiv_train-val_windowsz-30.0_overlap-15.0.data.pkl
	mkdir -p $(@D)
	python -u src/models/train_model.py $< $@ tremor --classifier gp --random-state $(*F) > $@.txt 2>&1

## Final MLP model for bradykinesia
models/gp/%/bradykinesia_GENEActiv_train-val_windowsz-30.0_overlap-15.0.model.pkl: data/processed/sktime/GENEActiv_train-val_windowsz-30.0_overlap-15.0.data.pkl
	mkdir -p $(@D)
	python -u src/models/train_model.py $< $@ bradykinesia --classifier gp --random-state $(*F) > $@.txt 2>&1

## Final MLP model for dyskinesia
models/gp/%/dyskinesia_GENEActiv_train-val_windowsz-30.0_overlap-15.0.model.pkl: data/processed/sktime/GENEActiv_train-val_windowsz-30.0_overlap-15.0.data.pkl
	mkdir -p $(@D)
	python -u src/models/train_model.py $< $@ dyskinesia --classifier gp --random-state $(*F) > $@.txt 2>&1

## Final ROCKET model for tremor
models/rocket/%/tremor_GENEActiv_train-val_windowsz-30.0_overlap-15.0.model.pkl: data/processed/sktime/GENEActiv_train-val_windowsz-30.0_overlap-15.0.data.pkl
	mkdir -p $(@D)
	python -u src/models/train_model.py $< $@ tremor --classifier rocket --random-state $(*F) > $@.txt 2>&1

## Final ROCKET model for bradykinesia
models/rocket/%/bradykinesia_GENEActiv_train-val_windowsz-30.0_overlap-15.0.model.pkl: data/processed/sktime/GENEActiv_train-val_windowsz-30.0_overlap-15.0.data.pkl
	mkdir -p $(@D)
	python -u src/models/train_model.py $< $@ bradykinesia --classifier rocket --random-state $(*F) > $@.txt 2>&1

## Final ROCKET model for dyskinesia
models/rocket/%/dyskinesia_GENEActiv_train-val_windowsz-30.0_overlap-15.0.model.pkl: data/processed/sktime/GENEActiv_train-val_windowsz-30.0_overlap-15.0.data.pkl
	mkdir -p $(@D)
	python -u src/models/train_model.py $< $@ dyskinesia --classifier rocket --random-state $(*F) > $@.txt 2>&1

## Final InceptionTime default model for tremor
models/inceptiontime/%/tremor_Shimmer-Wrist_train-val_windowsz-30.0_overlap-15.0.model.h5: data/processed/sktime/Shimmer-Wrist_train-val_windowsz-30.0_overlap-15.0.data.pkl
	mkdir -p $(@D)
	python -u src/models/train_model.py $< $@ tremor --classifier inceptiontime --epochs 600 --random-state $(*F) --no-checkpointing > $@.txt 2>&1

## Final InceptionTime default model for bradykinesia
models/inceptiontime/%/bradykinesia_Shimmer-Wrist_train-val_windowsz-30.0_overlap-15.0.model.h5: data/processed/sktime/Shimmer-Wrist_train-val_windowsz-30.0_overlap-15.0.data.pkl
	mkdir -p $(@D)
	python -u src/models/train_model.py $< $@ bradykinesia --classifier inceptiontime --epochs 600 --random-state $(*F) --no-checkpointing > $@.txt 2>&1

## Final InceptionTime default model for dyskinesia
models/inceptiontime/%/dyskinesia_Shimmer-Wrist_train-val_windowsz-30.0_overlap-15.0.model.h5: data/processed/sktime/Shimmer-Wrist_train-val_windowsz-30.0_overlap-15.0.data.pkl
	mkdir -p $(@D)
	python -u src/models/train_model.py $< $@ dyskinesia --classifier inceptiontime --epochs 600 --random-state $(*F) --no-checkpointing > $@.txt 2>&1

models/inceptiontimetuned/%/tremor_Shimmer-Wrist_train-val_windowsz-24.57_overlap-12.29.model.h5: data/processed/sktime/Shimmer-Wrist_train-val_windowsz-24.57_overlap-12.29.data.pkl
	mkdir -p $(@D)
	python -u src/models/train_model.py $< $@ tremor --classifier inceptiontime --epochs 600 --random-state $(*F) --use-best-params --no-checkpointing > $@.txt 2>&1

## Final InceptionTime model for dyskinesia incl. hyperparameter optimization
models/inceptiontimetuned/%/dyskinesia_Shimmer-Wrist_train-val_windowsz-29.33_overlap-14.66.model.h5: data/processed/sktime/Shimmer-Wrist_train-val_windowsz-29.33_overlap-14.66.data.pkl
	mkdir -p $(@D)
	python -u src/models/train_model.py $< $@ dyskinesia --classifier inceptiontime --epochs 600 --random-state $(*F) --use-best-params --no-checkpointing > $@.txt 2>&1

models/inceptiontimetuned/%/bradykinesia_Shimmer-Wrist_train-val_windowsz-22.37_overlap-11.18.model.h5: data/processed/sktime/Shimmer-Wrist_train-val_windowsz-22.37_overlap-11.18.data.pkl
	mkdir -p $(@D)
	python -u src/models/train_model.py $< $@ bradykinesia --classifier inceptiontime --epochs 600 --random-state $(*F) --use-best-params --no-checkpointing > $@.txt 2>&1

## Final InceptionTime default model for tremor
models/inceptiontime/%/tremor_Shimmer-Wrist_train-val_windowsz-30.0_overlap-15.0.model.h5: data/processed/sktime/Shimmer-Wrist_train-val_windowsz-30.0_overlap-15.0.data.pkl
	mkdir -p $(@D)
	python -u src/models/train_model.py $< $@ tremor --classifier inceptiontime --epochs 600 --random-state $(*F) > $@.txt 2>&1

## Final MLP model for tremor
models/gp/%/tremor_Shimmer-Wrist_train-val_windowsz-30.0_overlap-15.0.model.pkl: data/processed/sktime/Shimmer-Wrist_train-val_windowsz-30.0_overlap-15.0.data.pkl
	mkdir -p $(@D)
	python -u src/models/train_model.py $< $@ tremor --classifier gp --random-state $(*F) > $@.txt 2>&1

## Final MLP model for bradykinesia
models/gp/%/bradykinesia_Shimmer-Wrist_train-val_windowsz-30.0_overlap-15.0.model.pkl: data/processed/sktime/Shimmer-Wrist_train-val_windowsz-30.0_overlap-15.0.data.pkl
	mkdir -p $(@D)
	python -u src/models/train_model.py $< $@ bradykinesia --classifier gp --random-state $(*F) > $@.txt 2>&1

## Final MLP model for dyskinesia
models/gp/%/dyskinesia_Shimmer-Wrist_train-val_windowsz-30.0_overlap-15.0.model.pkl: data/processed/sktime/Shimmer-Wrist_train-val_windowsz-30.0_overlap-15.0.data.pkl
	mkdir -p $(@D)
	python -u src/models/train_model.py $< $@ dyskinesia --classifier gp --random-state $(*F) > $@.txt 2>&1

## Final ROCKET model for tremor
models/rocket/%/tremor_Shimmer-Wrist_train-val_windowsz-30.0_overlap-15.0.model.pkl: data/processed/sktime/Shimmer-Wrist_train-val_windowsz-30.0_overlap-15.0.data.pkl
	mkdir -p $(@D)
	python -u src/models/train_model.py $< $@ tremor --classifier rocket --random-state $(*F) > $@.txt 2>&1

## Final ROCKET model for bradykinesia
models/rocket/%/bradykinesia_Shimmer-Wrist_train-val_windowsz-30.0_overlap-15.0.model.pkl: data/processed/sktime/Shimmer-Wrist_train-val_windowsz-30.0_overlap-15.0.data.pkl
	mkdir -p $(@D)
	python -u src/models/train_model.py $< $@ bradykinesia --classifier rocket --random-state $(*F) > $@.txt 2>&1

## Final ROCKET model for dyskinesia
models/rocket/%/dyskinesia_Shimmer-Wrist_train-val_windowsz-30.0_overlap-15.0.model.pkl: data/processed/sktime/Shimmer-Wrist_train-val_windowsz-30.0_overlap-15.0.data.pkl
	mkdir -p $(@D)
	python -u src/models/train_model.py $< $@ dyskinesia --classifier rocket --random-state $(*F) > $@.txt 2>&1

# Data

## Make the required sktime dataframes and pickle them
data/processed/sktime/%.data.pkl: data/external/ data/external_filelist.txt
# $(VERIFY_DATA_EXTERNAL_FILELIST)
	python src/data/create_sktime_data.py $< $@

## Download the raw data from Synapse
data/external/:
	python src/data/download_raw_data.py $@

## Summarize the downloaded data for reference in Git
data/external_filelist.txt: data/external/
	find $< $(FIND_ARGS) | sort | grep -vF .tsv > $@

## Install Python dependencies
requirements: test_environment
	python -m pip install -U pip setuptools wheel
	python -m pip install -r requirements.txt
	python -m pip install --editable .

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

.PHONY: docker-build
## Docker Build
docker-build: Dockerfile requirements.txt $(wildcard src/**/*)
	docker build -t mjff_ld:$(GIT_REVISION) -t mjff_ld --build-arg GIT_COMMIT=$(GIT_REVISION) .

.PHONY: docker-push-gcp
## Docker Push to GCP Registry
docker-push-gcp: docker-build
	docker tag mjff_ld:$(GIT_REVISION) $(GCP_DOCKER_URI):$(GIT_REVISION)
	docker tag mjff_ld $(GCP_DOCKER_URI)
	docker push -a $(GCP_DOCKER_URI)

.PHONY: sync-data-to-gcp
## Sync data to GCP
sync-data-to-gcp: $(patsubst %hyperparameters.json,%hyperparameters.json-deps,$(wildcard models/*/hyperparameter_tuning/hyperparameters.json))
	gsutil -m rsync -r data/processed gs://$(GCP_BUCKET_NAME)/data/processed
	gsutil rsync -r models/inceptiontime/hyperparameter_tuning gs://$(GCP_BUCKET_NAME)/models/inceptiontime/hyperparameter_tuning

# Hyperparameters
## Hyperparameter definition
models/inceptiontime/hyperparameter_tuning/hyperparameters.json: src/models/create_hyperparameter_sample.py
	python $< $@

## Datasets required for hyperparameter search (for a given hyperparameter json file). Recursive due to https://stackoverflow.com/a/66393092
%hyperparameters.json-deps: %hyperparameters.json
	$(MAKE) $(shell python src/models/makedepend_hyperparameters.py models/inceptiontime/hyperparameter_tuning/hyperparameters.json)

## Tune hyperparameters
models/inceptiontime/hyperparameter_tuning/%_j-0.json \
models/inceptiontime/hyperparameter_tuning/%_j-1.json \
models/inceptiontime/hyperparameter_tuning/%_j-2.json \
models/inceptiontime/hyperparameter_tuning/%_j-3.json:
#data/processed/sktime/$$(basename $$(*F)).data.pkl models/inceptiontime/hyperparameter_tuning/hyperparameters.json
	echo data/processed/sktime/$(basename $(*F)).data.pkl 
	@echo $(subst .vm-,,$(suffix $(*F)))
	parallel -j4 echo 'CUDA_VISIBLE_DEVICES=$$(({%} -1))' \
		python src/models/crossval_score.py \
		gs://$(GCP_BUCKET_NAME)/$< \
		gs://$(GCP_BUCKET_NAME)/$(word 2,$^) \
		$(subst .vm-,,$(suffix $(*F))) \
		{} \
		'gs://$(GCP_BUCKET_NAME)/$(basename $@)_j-$$(({%} -1)).json' \
		--vm-count 4 \
		--epochs 10 \
		::: $(shell seq 0 14)

.PHONY: update-gcp
## Update GCP with source code and data. Do this before training.
update-gcp: docker-push-gcp sync-data-to-gcp

## Build and save the docker container
build/mjff_ld.tar.gz: Dockerfile requirements.txt
	docker build -t mjff_ld:$(GIT_REVISION) -t mjff_ld .
	docker save -o $@ mjff_ld

## Set up python interpreter environment
create_environment:
	$(PYTHON_INTERPRETER) -m venv $(VENV)
	@echo "Created virtualenv in $(VENV)."
	@bash -c "source $(VENV)/bin/activate"

## Test python environment is setup correctly
test_environment:
	@which python
	python test_environment.py

# Reports
reports/figures/exampletremors.pgf: data/external/ src/visualization/raw_data.py
	python src/visualization/raw_data.py $@

models/inceptiontime/hyperparameter_tuning/%/GENEActiv_train-val/results.csv \
models/inceptiontime/hyperparameter_tuning/%/GENEActiv_train-val/meanresults.csv:
	python src/visualization/summarize_hyperparameter_tuning.py $(*F) $(@D)/results.csv $(@D)/meanresults.csv

reports/figures/inceptiontime/hyperparameter_tuning/%/GENEActiv_train-val/mAPoverall.pgf \
reports/figures/inceptiontime/hyperparameter_tuning/%/GENEActiv_train-val/baovermAP.pgf: notebooks/crossval_scoring.ipynb models/inceptiontime/hyperparameter_tuning/%/GENEActiv_train-val/results.csv models/inceptiontime/hyperparameter_tuning/%/GENEActiv_train-val/meanresults.csv
	papermill  --cwd notebooks/ $< -p RESULTS_PATH ../$(word 2,$^) -p MEAN_RESULTS_PATH ../$(word 3,$^) -p PHENOTYPE $* $<.tmp-papermill.ipynb
	rm $<.tmp-papermill.ipynb

reports/figures/hypertuningscatter.pgf: notebooks/crossval_scoring.ipynb
	papermill  --cwd notebooks/ $< $<.tmp-papermill.ipynb
	rm $<.tmp-papermill.ipynb

reports/data/%_GENEActiv_train-val_meanresults.csv: models/inceptiontime/hyperparameter_tuning/%/GENEActiv_train-val/meanresults.csv
	sed 's/_//g' $< > $@

reports/data/%_GENEActiv_train-val_meanresults_head.csv: reports/data/%_GENEActiv_train-val_meanresults.csv
	head --lines 11 $< > $@

reports/figures/inceptiontime_%_training.pgf: models/inceptiontime/%.model.h5.txt src/visualization/main.mplstyle
	python src/visualization/inceptiontime_training.py $< $@

reports/figures/%/earlystopping.pgf: models/inceptiontime/$$(*D).model.h5 data/processed/sktime/$$(*F).data.pkl src/visualization/main.mplstyle
	mkdir -p $(@D)
	python src/visualization/early_stopping.py $< $(*F) $@

reports/figures/inceptiontime_tremor_hyperparams_optimvsdefault.pgf: models/inceptiontime/tremor_GENEActiv_train_windowsz-30.0_overlap-15.0.model.h5.pred/GENEActiv_val_windowsz-30.0_overlap-24.0.metrics.json	models/inceptiontime/tremor_GENEActiv_train_windowsz-21.88_overlap-10.94.model.h5.pred/GENEActiv_val_windowsz-21.89_overlap-17.5.metrics.json 
	python src/visualization/compare_metrics.py $^ --metric-names default optimized --legend-name Hyperparameters --output $@

reports/data/dyskinesia_GENEActiv_samewindowsize_results.csv reports/data/dyskinesia_GENEActiv_samewindowsize_meanresults.csv:
	python src/visualization/summarize_hyperparameter_tuning.py \
		--subdirectory 7dbb32c4/ \
		dyskinesia \
		reports/data/dyskinesia_GENEActiv_samewindowsize_results.csv \
		reports/data/dyskinesia_GENEActiv_samewindowsize_meanresults.csv

reports/data/tremor_GENEActiv_samewindowsize_results.csv reports/data/tremor_GENEActiv_samewindowsize_meanresults.csv:
	python src/visualization/summarize_hyperparameter_tuning.py \
		--subdirectory 7dbb32c4/ \
		tremor \
		reports/data/tremor_GENEActiv_samewindowsize_results.csv \
		reports/data/tremor_GENEActiv_samewindowsize_meanresults.csv

reports/figures/inceptiontime/dyskinesiamAPoverwindowlength.pgf: reports/data/dyskinesia_GENEActiv_samewindowsize_results.csv reports/data/dyskinesia_GENEActiv_samewindowsize_meanresults.csv
	python src/visualization/hyperparameter_scatterplots.py $^ --window-length $@

reports/figures/inceptiontime/tremormAPoverwindowlength.pgf: reports/data/tremor_GENEActiv_windowonly_results.csv reports/data/tremor_GENEActiv_windowonly_meanresults.csv
	python src/visualization/hyperparameter_scatterplots.py $^ --window-length $@

DEPS_FINAL_TREMOR := $(shell echo models/{gp,rocket}/{0..9}/tremor_GENEActiv_train-val_windowsz-30.0_overlap-15.0.model.pkl.pred/GENEActiv_test_windowsz-30.0_overlap-24.0.metrics.json models/inceptiontime/{0..9}/tremor_GENEActiv_train-val_windowsz-30.0_overlap-15.0.model.h5.pred/GENEActiv_test_windowsz-30.0_overlap-24.0.metrics.json models/inceptiontimetuned/{0..9}/tremor_GENEActiv_train-val_windowsz-24.57_overlap-12.29.model.h5.pred/GENEActiv_test_windowsz-24.57_overlap-19.66.metrics.json)
DEPS_FINAL_DYSKINESIA := $(shell echo models/{gp,rocket}/{0..9}/dyskinesia_GENEActiv_train-val_windowsz-30.0_overlap-15.0.model.pkl.pred/GENEActiv_test_windowsz-30.0_overlap-24.0.metrics.json models/inceptiontime/{0..9}/dyskinesia_GENEActiv_train-val_windowsz-30.0_overlap-15.0.model.h5.pred/GENEActiv_test_windowsz-30.0_overlap-24.0.metrics.json models/inceptiontimetuned/{0..9}/dyskinesia_GENEActiv_train-val_windowsz-29.33_overlap-14.66.model.h5.pred/GENEActiv_test_windowsz-29.33_overlap-23.46.metrics.json)
DEPS_FINAL_BRADYKINESIA := $(shell echo models/{gp,rocket}/{0..9}/bradykinesia_GENEActiv_train-val_windowsz-30.0_overlap-15.0.model.pkl.pred/GENEActiv_test_windowsz-30.0_overlap-24.0.metrics.json models/inceptiontime/{0..9}/bradykinesia_GENEActiv_train-val_windowsz-30.0_overlap-15.0.model.h5.pred/GENEActiv_test_windowsz-30.0_overlap-24.0.metrics.json models/inceptiontimetuned/{0..9}/bradykinesia_GENEActiv_train-val_windowsz-22.37_overlap-11.18.model.h5.pred/GENEActiv_test_windowsz-22.37_overlap-17.89.metrics.json)
DEPS_FINAL_SHIMMER_TREMOR := $(shell echo models/{gp,rocket}/{0..9}/tremor_Shimmer-Wrist_train-val_windowsz-30.0_overlap-15.0.model.pkl.pred/Shimmer-Wrist_test_windowsz-30.0_overlap-24.0.metrics.json models/inceptiontime/{0..9}/tremor_Shimmer-Wrist_train-val_windowsz-30.0_overlap-15.0.model.h5.pred/Shimmer-Wrist_test_windowsz-30.0_overlap-24.0.metrics.json models/inceptiontimetuned/{0..9}/tremor_Shimmer-Wrist_train-val_windowsz-24.57_overlap-12.29.model.h5.pred/Shimmer-Wrist_test_windowsz-24.57_overlap-19.66.metrics.json)
DEPS_FINAL_SHIMMER_DYSKINESIA := $(shell echo models/{gp,rocket}/{0..9}/dyskinesia_Shimmer-Wrist_train-val_windowsz-30.0_overlap-15.0.model.pkl.pred/Shimmer-Wrist_test_windowsz-30.0_overlap-24.0.metrics.json models/inceptiontime/{0..9}/dyskinesia_Shimmer-Wrist_train-val_windowsz-30.0_overlap-15.0.model.h5.pred/Shimmer-Wrist_test_windowsz-30.0_overlap-24.0.metrics.json models/inceptiontimetuned/{0..9}/dyskinesia_Shimmer-Wrist_train-val_windowsz-29.33_overlap-14.66.model.h5.pred/Shimmer-Wrist_test_windowsz-29.33_overlap-23.46.metrics.json)
DEPS_FINAL_SHIMMER_BRADYKINESIA := $(shell echo models/{gp,rocket}/{0..9}/bradykinesia_Shimmer-Wrist_train-val_windowsz-30.0_overlap-15.0.model.pkl.pred/Shimmer-Wrist_test_windowsz-30.0_overlap-24.0.metrics.json models/inceptiontime/{0..9}/bradykinesia_Shimmer-Wrist_train-val_windowsz-30.0_overlap-15.0.model.h5.pred/Shimmer-Wrist_test_windowsz-30.0_overlap-24.0.metrics.json models/inceptiontimetuned/{0..9}/bradykinesia_Shimmer-Wrist_train-val_windowsz-22.37_overlap-11.18.model.h5.pred/Shimmer-Wrist_test_windowsz-22.37_overlap-17.89.metrics.json)

reports/figures/tremormodelcomparison.pgf: $(DEPS_FINAL_TREMOR)
	python src/visualization/evaluate_multiple_models.py --output-figure $@ $^

reports/figures/modelcomparison.pgf: $(DEPS_FINAL_TREMOR) $(DEPS_FINAL_BRADYKINESIA) $(DEPS_FINAL_DYSKINESIA)
	python src/visualization/evaluate_multiple_models.py --output-figure $@ $^

reports/figures/dyskinesiamodelcomparison.pgf: $(DEPS_FINAL_DYSKINESIA)
	python src/visualization/evaluate_multiple_models.py --output-figure $@ $^

reports/figures/bradykinesiamodelcomparison.pgf: $(DEPS_FINAL_BRADYKINESIA)
	python src/visualization/evaluate_multiple_models.py --output-figure $@ $^

reports/figures/inceptiontime/tremormAPoverwindowlength.pgf: reports/data/tremor_GENEActiv_windowonly_results.csv reports/data/tremor_GENEActiv_windowonly_meanresults.csv
	python src/visualization/hyperparameter_scatterplots.py $^ --window-length $@

reports/figures/shimmertremormodelcomparison.pgf: $(DEPS_FINAL_SHIMMER_TREMOR)
	python src/visualization/evaluate_multiple_models.py --output-figure $@ $^

reports/figures/shimmerdyskinesiamodelcomparison.pgf: $(DEPS_FINAL_SHIMMER_DYSKINESIA)
	python src/visualization/evaluate_multiple_models.py --output-figure $@ $^

reports/figures/shimmerbradykinesiamodelcomparison.pgf: $(DEPS_FINAL_SHIMMER_BRADYKINESIA)
	python src/visualization/evaluate_multiple_models.py --output-figure $@ $^

reports/figures/shimmermodelcomparison.pgf: $(DEPS_FINAL_SHIMMER_TREMOR) $(DEPS_FINAL_SHIMMER_BRADYKINESIA) $(DEPS_FINAL_SHIMMER_DYSKINESIA)
	python src/visualization/evaluate_multiple_models.py --output-figure $@ $^

models/gp/%_summary.txt: SHELL:=/bin/bash
models/gp/%_summary.txt: $$(shell echo models/gp/{0..9}/$$(*F)_GENEActiv_train-val_windowsz-30.0_overlap-15.0.model.pkl.pred/GENEActiv_test_windowsz-30.0_overlap-24.0.metrics.json)
	python src/visualization/evaluate_multiple_models.py $^ > $@

models/rocket/%_summary.txt: SHELL:=/bin/bash
models/rocket/%_summary.txt: $$(shell echo models/rocket/{0..9}/$$(*F)_GENEActiv_train-val_windowsz-30.0_overlap-15.0.model.pkl.pred/GENEActiv_test_windowsz-30.0_overlap-24.0.metrics.json)
	python src/visualization/evaluate_multiple_models.py $^ > $@

reports/data/tremor_models.csv: $(DEPS_FINAL_TREMOR) $(DEPS_FINAL_SHIMMER_TREMOR)
	python src/visualization/evaluate_multiple_models.py --output-csv $@ $^

reports/data/bradykinesia_models.csv: $(DEPS_FINAL_DYSKINESIA) $(DEPS_FINAL_SHIMMER_DYSKINESIA)
	python src/visualization/evaluate_multiple_models.py --output-csv $@ $^

reports/data/dyskinesia_models.csv: $(DEPS_FINAL_BRADYKINESIA) $(DEPS_FINAL_SHIMMER_BRADYKINESIA)
	python src/visualization/evaluate_multiple_models.py --output-csv $@ $^

reports/include/gen/fulldata.tex: data/external/
	python src/visualization/data_visualization.py --full-data $@

reports/include/gen/trainvsval.tex: data/external/
	python src/visualization/data_visualization.py --train-vs-val $@

reports/include/gen/tvvstest.tex: data/external/
	python src/visualization/data_visualization.py --trainval-vs-test $@

reports/figures/phenotypedistribution.pgf: data/external/
	python src/visualization/data_visualization.py --figure $@

reports/figures/inceptiontimetremormedian/confusionmatrix.pgf: models/inceptiontime/9/tremor_GENEActiv_train-val_windowsz-30.0_overlap-15.0.model.h5.pred/GENEActiv_test_windowsz-30.0_overlap-24.0.pred.npz
	python src/visualization/metrics.py --output-path $(@D) $<

reports/figures/gptremormedian/confusionmatrix.pgf: models/gp/8/tremor_GENEActiv_train-val_windowsz-30.0_overlap-15.0.model.pkl.pred/GENEActiv_test_windowsz-30.0_overlap-24.0.pred.npz
	python src/visualization/metrics.py --output-path $(@D) $<

reports/figures/rockettremormedian/confusionmatrix.pgf: models/rocket/5/tremor_GENEActiv_train-val_windowsz-30.0_overlap-15.0.model.pkl.pred/GENEActiv_test_windowsz-30.0_overlap-24.0.pred.npz
	python src/visualization/metrics.py --output-path $(@D) $<

reports/figures/inceptiontimetunedtremormedian/confusionmatrix.pgf: models/inceptiontimetuned/1/tremor_GENEActiv_train-val_windowsz-24.57_overlap-12.29.model.h5.pred/GENEActiv_test_windowsz-24.57_overlap-19.66.pred.npz
	python src/visualization/metrics.py --output-path $(@D) $<

reports/figures/inceptiontimedyskinesiamedian/confusionmatrix.pgf: models/inceptiontime/7/dyskinesia_GENEActiv_train-val_windowsz-30.0_overlap-15.0.model.h5.pred/GENEActiv_test_windowsz-30.0_overlap-24.0.pred.npz
	python src/visualization/metrics.py --output-path $(@D) $<

reports/figures/gpdyskinesiamedian/confusionmatrix.pgf: models/gp/7/dyskinesia_GENEActiv_train-val_windowsz-30.0_overlap-15.0.model.pkl.pred/GENEActiv_test_windowsz-30.0_overlap-24.0.pred.npz
	python src/visualization/metrics.py --output-path $(@D) $<

reports/figures/rocketdyskinesiamedian/confusionmatrix.pgf: models/rocket/3/dyskinesia_GENEActiv_train-val_windowsz-30.0_overlap-15.0.model.pkl.pred/GENEActiv_test_windowsz-30.0_overlap-24.0.pred.npz
	python src/visualization/metrics.py --output-path $(@D) $<

reports/figures/inceptiontimetuneddyskinesiamedian/confusionmatrix.pgf: models/inceptiontimetuned/6/dyskinesia_GENEActiv_train-val_windowsz-29.33_overlap-14.66.model.h5.pred/GENEActiv_test_windowsz-29.33_overlap-23.46.pred.npz
	python src/visualization/metrics.py --output-path $(@D) $<

reports/figures/inceptiontimebradykinesiamedian/confusionmatrix.pgf: models/inceptiontime/7/bradykinesia_GENEActiv_train-val_windowsz-30.0_overlap-15.0.model.h5.pred/GENEActiv_test_windowsz-30.0_overlap-24.0.pred.npz
	python src/visualization/metrics.py --output-path $(@D) $<

reports/figures/gpbradykinesiamedian/confusionmatrix.pgf: models/gp/9/bradykinesia_GENEActiv_train-val_windowsz-30.0_overlap-15.0.model.pkl.pred/GENEActiv_test_windowsz-30.0_overlap-24.0.pred.npz
	python src/visualization/metrics.py --output-path $(@D) $<

reports/figures/rocketbradykinesiamedian/confusionmatrix.pgf: models/rocket/5/bradykinesia_GENEActiv_train-val_windowsz-30.0_overlap-15.0.model.pkl.pred/GENEActiv_test_windowsz-30.0_overlap-24.0.pred.npz
	python src/visualization/metrics.py --output-path $(@D) $<

reports/figures/inceptiontimetunedbradykinesiamedian/confusionmatrix.pgf: models/inceptiontimetuned/4/bradykinesia_GENEActiv_train-val_windowsz-22.37_overlap-11.18.model.h5.pred/GENEActiv_test_windowsz-22.37_overlap-17.89.pred.npz
	python src/visualization/metrics.py --output-path $(@D) $<


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http:$(eval foo=$(shell ls | grep makefile))
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \epComparison
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
