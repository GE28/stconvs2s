import sys
from netCDF4 import Dataset

if len(sys.argv) < 2:
    print("Usage: python analyze_nc.py <path_to_nc_file>")
    sys.exit(1)

nc_path = sys.argv[1]

try:
    ds = Dataset(nc_path, 'r')
except Exception as e:
    print(f"Error opening file: {e}")
    sys.exit(1)

print(f"\nFile: {nc_path}\n")

print("Dimensions:")
for dim in ds.dimensions.values():
    print(f"  {dim.name}: {len(dim)}")

print("\nVariables:")
for var in ds.variables.values():
    print(f"  {var.name}: {var.dimensions} {var.shape}")

print("\nGlobal Attributes:")
for attr in ds.ncattrs():
    print(f"  {attr}: {getattr(ds, attr)}")

print("\nCoordinates (variables with same name as dimension):")
for dim in ds.dimensions:
    if dim in ds.variables:
        print(f"  {dim}")

ds.close()
