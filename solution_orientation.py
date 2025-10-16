import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np
from shapely.geometry import Point, LineString

# Define input and output paths
input_directory = Path.cwd() / 'input/'
output_directory = Path.cwd() / 'output/'

# Load cadastre (properties) data
cadastre_path = input_directory / 'cadastre.gpkg'
cadastre_gdf = gpd.read_file(str(cadastre_path), layer=0)

# Load road data
roads_path = input_directory / 'roads.gpkg'
roads_gdf = gpd.read_file(str(roads_path), layer=0)

# Load property information with addresses
gnaf_df = pd.read_parquet(input_directory / 'gnaf_prop.parquet')

# Convert gnaf points to GeoDataFrame
gnaf_gdf = gpd.GeoDataFrame(
    gnaf_df, 
    geometry=gpd.points_from_xy(gnaf_df.longitude, gnaf_df.latitude),
    crs="EPSG:4326"  # Assuming WGS84 coordinates
)

# Convert all to a projected CRS for accurate distance calculations - using GDA2020 MGA Zone 56
# This is suitable for eastern Australia where Sydney is located
projected_crs = "EPSG:7856"
cadastre_gdf = cadastre_gdf.to_crs(projected_crs)
roads_gdf = roads_gdf.to_crs(projected_crs)
gnaf_gdf = gnaf_gdf.to_crs(projected_crs)

# Function to determine the orientation based on the property's position relative to the nearest road
def determine_orientation(property_geom, nearest_road_geom):
    # Calculate the centroid of the property
    property_centroid = property_geom.centroid
    
    # For properties with MultiPolygon geometry, we'll focus on the largest polygon
    if property_geom.geom_type == 'MultiPolygon':
        areas = [p.area for p in property_geom.geoms]
        largest_poly_idx = areas.index(max(areas))
        property_geom = property_geom.geoms[largest_poly_idx]
    
    # For simple cases, we'll use the angle between the property centroid and the nearest point on the road
    if isinstance(nearest_road_geom, LineString):
        # Find nearest point on the road to the property centroid
        nearest_point = nearest_road_geom.interpolate(nearest_road_geom.project(property_centroid))
        
        # Calculate the angle between the property centroid and the nearest point
        angle = np.degrees(np.arctan2(
            nearest_point.y - property_centroid.y, 
            nearest_point.x - property_centroid.x
        ))
        
        # Convert angle to compass direction
        # Angle is measured in degrees, where 0 is East, 90 is North, 180/-180 is West, -90 is South
        if angle < -157.5 or angle >= 157.5:
            return "West"
        elif angle < -112.5:
            return "Southwest"
        elif angle < -67.5:
            return "South"
        elif angle < -22.5:
            return "Southeast"
        elif angle < 22.5:
            return "East"
        elif angle < 67.5:
            return "Northeast"
        elif angle < 112.5:
            return "North"
        elif angle < 157.5:
            return "Northwest"
    
    # Default case if we can't determine orientation
    return "Unknown"

# Create spatial index for roads to speed up nearest road calculations
roads_sindex = roads_gdf.sindex

# Create a results dataframe to store property information and orientation
results = []

# Process each property in the cadastre
for idx, property_row in cadastre_gdf.iterrows():
    property_geom = property_row.geometry
    
    # Find the nearest road
    potential_road_indices = list(roads_sindex.intersection(property_geom.bounds))
    if potential_road_indices:
        # Get nearby roads
        nearby_roads = roads_gdf.iloc[potential_road_indices]
        
        # Calculate distances to each road
        distances = nearby_roads.geometry.distance(property_geom)
        
        # Get the closest road
        nearest_road = nearby_roads.iloc[distances.argmin()]
        nearest_road_geom = nearest_road.geometry
        
        # Determine orientation
        orientation = determine_orientation(property_geom, nearest_road_geom)
        
        # Calculate parameters for neural network training
        property_centroid = property_geom.centroid
        if property_geom.geom_type == 'MultiPolygon':
            areas = [p.area for p in property_geom.geoms]
            largest_poly_idx = areas.index(max(areas))
            property_geom = property_geom.geoms[largest_poly_idx]
        
        nearest_point = None
        angle = None
        if isinstance(nearest_road_geom, LineString):
            # Find nearest point on the road to the property centroid
            nearest_point = nearest_road_geom.interpolate(nearest_road_geom.project(property_centroid))
            # Calculate the angle between the property centroid and the nearest point
            angle = np.degrees(np.arctan2(
                nearest_point.y - property_centroid.y, 
                nearest_point.x - property_centroid.x
            ))
        
        # Try to find address for this property by spatial join with gnaf data
        matching_addresses = gnaf_gdf[gnaf_gdf.within(property_geom)]
        
        # Common parameters for both cases
        nn_params = {
            'property_centroid_x': property_centroid.x if property_centroid else None,
            'property_centroid_y': property_centroid.y if property_centroid else None,
            'property_area': property_geom.area,
            'property_perimeter': property_geom.length,
            'nearest_point_x': nearest_point.x if nearest_point else None,
            'nearest_point_y': nearest_point.y if nearest_point else None,
            'angle_to_road': angle,
            'road_id': nearest_road.name if hasattr(nearest_road, 'name') else idx,
            'distance_to_road': property_geom.distance(nearest_road_geom) if nearest_road_geom else None,
        }
        
        # If we found matching addresses, add them to our results
        if not matching_addresses.empty:
            for _, addr_row in matching_addresses.iterrows():
                entry = {
                    'address': addr_row['address'],
                    'suburb': addr_row['locality_name'],
                    'postcode': addr_row['postcode'],
                    'state': addr_row['state'],
                    'orientation': orientation,
                    'gnaf_pid': addr_row['gnaf_pid']  # Add GNAF ID for deduplication
                }
                # Add neural network parameters
                entry.update(nn_params)
                results.append(entry)
        else:
            # If no address found, still record the property with minimal information
            entry = {
                'address': f"Unknown Address (ID: {idx})",
                'suburb': "",
                'postcode': "",
                'state': "",
                'orientation': orientation,
                'gnaf_pid': ""  # Empty GNAF ID for unknown addresses
            }
            # Add neural network parameters
            entry.update(nn_params)
            results.append(entry)

# Create dataframe from results
orientation_df = pd.DataFrame(results)

# Remove duplicate addresses (some properties may have multiple points)
orientation_df = orientation_df.drop_duplicates(subset=['gnaf_pid'], keep='first')

# Create a copy of the dataframe with neural network parameters
nn_params_df = orientation_df.copy()

# Remove the GNAF ID column from final output
orientation_df = orientation_df.drop(columns=['gnaf_pid'])

# For the standard output, remove the NN parameters
nn_columns = [
    'property_centroid_x', 'property_centroid_y', 'property_area', 'property_perimeter',
    'nearest_point_x', 'nearest_point_y', 'angle_to_road', 'road_id', 'distance_to_road'
]
orientation_df = orientation_df.drop(columns=nn_columns, errors='ignore')

# Save to output file - entire dataset
orientation_df.to_csv(output_directory / 'property_orientations.csv', index=False)

# Save neural network parameters to a separate file
nn_params_file = output_directory / 'orientation_nn_parameters.csv'
nn_params_df.to_csv(nn_params_file, index=False)

print(f"Saved property orientations to {output_directory / 'property_orientations.csv'}")
print(f"Saved neural network parameters to {nn_params_file}")
print(f"Found orientations for {len(orientation_df)} properties")
