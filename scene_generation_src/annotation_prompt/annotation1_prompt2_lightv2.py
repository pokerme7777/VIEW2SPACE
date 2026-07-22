annotation1_prompt2="""
You are viewing an OVERVIEW image containing a collection of 3D assets.
You are provided with:
1. 'tag_library_0' (Base Names)
2. 'comparison_keys' (The list of specific visual attributes that distinguish these assets, e.g., "color", "hat style", "number of floors").

Your goal is to generate tags for EVERY single asset by **inspecting the specific attributes listed in 'comparison_keys'**.

# 1. VISUAL GROUNDING RULES
All tags must be based strictly on what is visible.

# 2. BASE NAME & SIMPLICITY
- Use simple, everyday vocabulary (e.g., use "car" not "automobile").
- If an object is a specific identifiable type (e.g., "Mercedes", "Coke bottle"), use that as the Base Name.
- If generic, use the standard category name.

# 3. TAG LIBRARY CONSTRUCTION

## tag_library_1: ATTRIBUTE DECOMPOSITION (Base Name + SINGLE features)
Goal: Decompose every object into its atomic attributes, **specifically focusing on the 'comparison_keys'**.

For EACH object, generate multiple tags.
**MANDATORY:** You MUST generate a tag for every attribute mentioned in 'comparison_keys' if it is visible.

**Construction Rule:**
- Iterate through ALL attributes in 'comparison_keys'.
- If an attribute is visible on the object, generate a **SEPARATE** tag for it.
- **Strict Constraint:** Each tag must contain **ONLY ONE** attribute + Base Name. Do NOT combine features here.

**Format:**
- '[Base Name] + [Feature A]'
- '[Base Name] + [Feature B]'

**Example:**
- If 'comparison_keys' = ["color", "hat style"]
- Object is a "Red Snowman with Top Hat".
- You can generate: '"red snowman"', '"snowman with top hat"'.
- (You can also add other visible features like '"snowman with buttons"').


## tag_library_2: GUIDED UNIQUE IDENTIFIER (The "Fastest Distinction" Rule)
Goal: Produce **exactly one unique identifying tag for each asset**, selecting the *simplest adequate distinguishing attribute(s)* from 'comparison_keys'.

This must follow the Differentiation Hierarchy below.

### DIFFERENTIATION HIERARCHY (Filter the 'comparison_keys' using this order)

PRIORITY 1 — COLOR / SHAPE / GEOMETRY  
• If 'comparison_keys' includes any of these, start here.  
• If this attribute alone distinguishes the object, stop.  
• Example: Keys=["color"], objects are red vs blue → use "red chair".

PRIORITY 2 — STRUCTURE / COMPONENTS / COUNT / SIZE  
• Use structural attributes if Priority 1 is insufficient (e.g., multiple red chairs).  
• Example: Keys=["color", "armrests"], both chairs are red , then use  
  "red chair with armrests" vs "red chair without armrests".

PRIORITY 3 — DETAIL / ACCESSORY / BRANDING  
• Use only if the object is identical in color AND structure.  
• Example: "white Starbucks cup" vs "white cup without logo".

PRIORITY 4 — ACTION / POSTURE (for characters)  
• Use only if earlier priorities fail to distinguish characters.  
• Example: “man without hat and sitting”.

tag_library_1 could only use one or two attribute from 'comparison_keys' for each tag, but tag_library_2 can combine multiple attributes from 'comparison_keys' if needed to ensure uniqueness.

**Constraint FOR tag_library_2:**
MAINLY use ** attributes present in 'comparison_keys'.** 
Do NOT over-describe. 
Every object in the image must have a corresponding entry in 'tag_library_2'. 
Do not group them. List them individually.

Return ONLY via the function call with this JSON structure:
{
  "tag_library_1": ["tag1", "tag2", ...],
  "tag_library_2": ["adaptive_unique_tag_1", "adaptive_unique_tag_2", ...]
}
"""