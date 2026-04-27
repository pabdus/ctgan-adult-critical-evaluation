## Data preprocessing

The preprocessing pipeline (in `src/preprocessing.py`) performs:
1. Replace `'?'` with NaN and drop affected rows (~7.4% of data)
2. Drop `fnlwgt` column (sampling weight, not an individual attribute)
3. Stratified sampling of 10,000 rows preserving income distribution
4. Hold out remaining ~20,000 rows for TSTR evaluation
