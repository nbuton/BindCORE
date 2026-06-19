#!/usr/bin/env bash
set -euo pipefail

echo "=================================================="
echo "LIP — IDPFold2"
echo "=================================================="
python3 scripts/attribution.py \
  --model data/models/BindCORE_LIP_IDPFold2/bindCORE.pt \
  --dataset data/LIP_dataset/TE440_less_than_1024.txt \
  --h5 data/properties/IDPFold2_derived_properties.h5 \
  --device cuda \
  --output data/interpretability/BindCORE_LIP_IDPFold2/feature_importance.csv

echo "=================================================="
echo "LIP — AF-CALVADOS"
echo "=================================================="
python3 scripts/attribution.py \
  --model data/models/BindCORE_LIP_AF_CALVADOS/bindCORE.pt \
  --dataset data/LIP_dataset/TE440_less_than_1024.txt \
  --h5 data/properties/single_AF_CALVADOS_derived_properties.h5 \
  --device cuda \
  --output data/interpretability/BindCORE_LIP_AF_CALVADOS/feature_importance.csv

echo "=================================================="
echo "LIP — STARLING"
echo "=================================================="
python3 scripts/attribution.py \
  --model data/models/BindCORE_LIP_STARLING/bindCORE.pt \
  --dataset data/LIP_dataset/TE440_less_than_380.txt \
  --h5 data/properties/STARLING_derived_properties.h5 \
  --device cuda \
  --output data/interpretability/BindCORE_LIP_STARLING/feature_importance.csv

echo "=================================================="
echo "MoRF — IDPFold2"
echo "=================================================="
python3 scripts/attribution.py \
  --model data/models/BindCORE_MoRF_IDPFold2/bindCORE.pt \
  --dataset data/MoRF_dataset/test.txt \
  --h5 data/properties/IDPFold2_derived_properties.h5 \
  --device cuda \
  --output data/interpretability/BindCORE_MoRF_IDPFold2/feature_importance.csv

echo "=================================================="
echo "MoRF — AF-CALVADOS"
echo "=================================================="
python3 scripts/attribution.py \
  --model data/models/BindCORE_MoRF_AF_CALVADOS/bindCORE.pt \
  --dataset data/MoRF_dataset/test.txt \
  --h5 data/properties/single_AF_CALVADOS_derived_properties.h5 \
  --device cuda \
  --output data/interpretability/BindCORE_MoRF_AF_CALVADOS/feature_importance.csv

echo "=================================================="
echo "MoRF — STARLING"
echo "=================================================="
python3 scripts/attribution.py \
  --model data/models/BindCORE_MoRF_STARLING/bindCORE.pt \
  --dataset data/MoRF_dataset/test_less_than_380.txt \
  --h5 data/properties/STARLING_derived_properties.h5 \
  --device cuda \
  --output data/interpretability/BindCORE_MoRF_STARLING/feature_importance.csv

echo "=================================================="
echo "All attribution runs completed successfully."
echo "=================================================="