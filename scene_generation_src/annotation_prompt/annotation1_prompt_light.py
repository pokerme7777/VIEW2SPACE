annotation1_prompt="""
You see an overview image containing several 3D asset previews from the same folder.

Your task is to analyze these assets and generate two things:
1. A list of basic category names (Base Names).
2. A list of "Comparison Keys" that define the **dimensions of difference** between these objects.

The goal is to build a flexible, reusable tag universe that can later describe any asset precisely.

# Core Principles
## 1. Everything must be visually grounded
## 2. Collective Discriminability Requirement


## 3. Tag Construction Guidelines
### Zero-feature tags are pure basic names or general category labels.

### They represent the core identity of an asset without any attributes.

### They must be very general, non-specific, and attribute-free.

### They should NOT try to distinguish between different individual objects.

### It is completely acceptable — even expected — that many or all objects share the same zero-feature tag.

### Examples:
  - "chair"
  - "mug"
  - "orange"
  - "laptop"
  - "toy"
  - "bottle"
  - "sofa"
  - "cabinet"
  - "backpack"


# 5. VISUAL VARIANCE ANALYSIS (For comparison_keys)
You must scan the collection and determine **which visual attributes vary the most** across the assets.
Do not just list standard attributes. List the specific attributes that change in THIS image.

### Search Strategy for Different Asset Types:
- **For Architecture (Houses, Stations, Shelters):**
  - Look for **Structural Structure**: "number of floors", "roof shape", "window style", "presence of garage/porch".
  - *Example:* If houses are same color but different sizes -> Key: "number of stories".

- **For Vehicles/Machines (Airplanes, Cars, Balloons):**
  - Look for **Sub-types & Attachments**: "wing shape", "propeller count", "basket type", "balloon pattern", "branding".
  - *Example:* If hot air balloons have different stripes -> Key: "balloon pattern".

- **For Simple Shapes (Balls, Crates, Generic Props):**
  - Look for **Surface & State**: "texture pattern", "color", "damage state", "open vs closed".

- **For Characters/Organics:**
  - Look for **Accessories & Pose**: "clothing color", "held item", "posture".

### Output Logic Example:
- **Scene: A mix of different houses.**
  - **comparison_keys:** ["number of floors", "roof style", "color of walls"]

- **Scene: Several hot air balloons with different patterns.**
  - **comparison_keys:** ["balloon pattern", "color scheme", "basket size"]

- **Scene: A group of airplanes (some jets, some biplanes).**
  - **comparison_keys:** ["wing configuration", "engine type", "livery color"]

- **Scene: Bus stops (some with benches, some with ads).**
  - **comparison_keys:** ["presence of bench", "advertisement poster", "shelter shape"]

Return ONLY via the function call with this JSON structure:
{
  "tag_library_0": ["tag1", "tag2", ...],
  "comparison_keys": ["feature_that_varies_1", "feature_that_varies_2", ...]
}

"""