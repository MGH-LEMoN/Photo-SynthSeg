PROJ_DIR := $(shell pwd)
DATA_DIR := $(PROJ_DIR)/data

remove-subject-copies:
	rm $(DATA_DIR)/SynthSeg_label_maps_manual_auto_photos/subject*copy*

create-subject-copies:
	python scripts/make_data_copy.py