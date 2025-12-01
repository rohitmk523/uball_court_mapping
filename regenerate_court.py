from app.services import dxf_parser
from pathlib import Path

print("Regenerating court image with 3-point lines...")

output_path = Path("data/calibration/court_image.png")
dxf_path = Path("court_2.dxf")

dxf_parser.generate_court_image(dxf_path, output_path, margin=200.0, scale=2.0)

print(f"Court image regenerated: {output_path}")
print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
