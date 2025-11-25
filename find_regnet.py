from torchvision import models

# Check different RegNet variants
regnets = [
    ('regnet_y_400mf', models.regnet_y_400mf),
    ('regnet_y_800mf', models.regnet_y_800mf),
    ('regnet_y_1_6gf', models.regnet_y_1_6gf),
    ('regnet_y_3_2gf', models.regnet_y_3_2gf),
    ('regnet_y_8gf', models.regnet_y_8gf),
    ('regnet_y_16gf', models.regnet_y_16gf),
    ('regnet_y_32gf', models.regnet_y_32gf),
]

print("Searching for RegNet with 1296 input features...")
print("="*60)

for name, model_fn in regnets:
    model = model_fn(weights=None)
    fc_features = model.fc.in_features
    print(f"{name:20s}: {fc_features:5d} features")
    
    if fc_features == 1296:
        print(f"  ✓ MATCH FOUND!")
