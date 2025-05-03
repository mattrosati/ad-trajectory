#!/bin/bash

# Check if path prefix is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <absolute path to raw data directory>"
  exit 1
fi

PREFIX="$1"


python ./src/data_clean.py --data_path ${PREFIX}/raw_data.nosync/ADNIMERGE/ADNIMERGE_30Apr2025.csv --vitals_path ${PREFIX}/raw_data.nosync/All_Subjects_VITALS_29Apr2025.csv --design_features 'log'  --output_path ${PREFIX}/processed/
python ./src/data_clean.py --data_path ${PREFIX}/raw_data.nosync/ADNIMERGE/ADNIMERGE_30Apr2025.csv --vitals_path ${PREFIX}/raw_data.nosync/All_Subjects_VITALS_29Apr2025.csv --design_features 'yeo-johnson'  --output_path ${PREFIX}/processed/
python ./src/data_clean.py --data_path ${PREFIX}/raw_data.nosync/ADNIMERGE/ADNIMERGE_30Apr2025.csv --vitals_path ${PREFIX}/raw_data.nosync/All_Subjects_VITALS_29Apr2025.csv --output_path ${PREFIX}/processed/
python ./src/data_clean.py --data_path ${PREFIX}/raw_data.nosync/ADNIMERGE/ADNIMERGE_30Apr2025.csv --output_path ${PREFIX}/processed/