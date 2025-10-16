# Property Orientation Analysis

## Overview
This project analyzes property orientations by determining which direction properties face (north, south, east, west, etc.) based on their geometric relationship to nearby roads.

## Approach
First inspected the data to understand property and road relationships. Then calculated orientations by finding each property's nearest road and determining the compass direction based on the geometric relationship between property centroid and road alignment.

## Findings
- Analyzed 1,198 properties with clear orientations
- East-facing properties were most common (201)
- South-facing properties were second most common (196)
- North-facing properties were also significant (156)

## Usage
1. Ensure dependencies are installed:
   ```
   pip install pandas geopandas numpy shapely
   ```

   if running NN example, more dependencies are needed
   ```
   pip install scikit-lear matplotlib seaborn tensorflow
   ```

2. Run the script:
   ```
   python solution_orientation.py
   ```

   Or execute the notebook.

## Output
Two output files are generated:
- `property_orientations.csv` - Complete dataset of property orientations

## Real Estate Implications
North-facing properties typically attract premium prices due to maximum natural light exposure, especially in the Southern Hemisphere. This data helps investors identify valuable properties based on orientation preferences.