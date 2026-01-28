"

import sys
import yaml
from ultralytics import YOLO



model_path = "models/sgm.pt"
config_path = "config/config.yaml"

# Load YOLO model
print(f"Loading model: {model_path}")
model = YOLO(model_path)

# Get class names
class_names = list(model.names.values())
print(f"Found {len(class_names)} classes:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

# Load existing config
print(f"\nUpdating {config_path}...")
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

# Update class names
cfg['class_names'] = class_names

# Write back
with open(config_path, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

print("Done!")

