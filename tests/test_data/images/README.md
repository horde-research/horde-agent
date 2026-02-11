# Test Images Directory

Place your test images here (`.jpg`, `.jpeg`, `.png`, or `.webp` files).

The test script will:
1. Load all images from this directory (recursively)
2. Annotate them using the LLM
3. Generate SFT training examples
4. Save results to `test_outputs/`

Example structure:
```
test_data/images/
  ├── street_scene.jpg
  ├── traditional_ceremony/
  │   ├── ceremony_1.jpg
  │   └── ceremony_2.jpg
  └── architecture/
      └── building_1.png
```

The folder names (e.g., `traditional_ceremony`) will be used as `topic_hint` in the annotation prompt.
