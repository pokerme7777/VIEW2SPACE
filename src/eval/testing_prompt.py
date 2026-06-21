counting_testing_prompt = """
Answer the counting question based on the provided image(s). 
"""

detection_testing_prompt = """
Answer the detection question based on the provided image(s). 
"""

MCQ_testing_prompt = """
Answer the MCQ question based on the provided image(s). 
"""

detection_normal_testing_prompt = """
Answer the detection question based on the provided image(s). You may reason step by step before answering.\n
After finishing your reasoning, output the final answer EXACTLY in the following format:
<answer> [[x1, y1, x2, y2], ...] </answer>

Each bounding box must use xyxy format with integer coordinates scaled to a 0-1000 range relative to the image.
If no target instances exist, output:
<answer> [] </answer>
"""

detection_normal_direct_testing_prompt = """
Answer the detection question based on the provided image(s). You need to directly answer question.\n
Output the final answer EXACTLY in the following format:
<answer> [[x1, y1, x2, y2], ...] </answer>

Each bounding box must use xyxy format with integer coordinates scaled to a 0-1000 range relative to the image.
If no target instances exist, output:
<answer> [] </answer>
"""

MCQ_normal_testing_prompt = """
Answer the MCQ question based on the provided image(s). You may reason step by step before answering.\n
After you finish your reasoning, output the final answer in the following format ONLY: <answer> A </answer>\n
The answer must be a single letter: A, B, C, or D.
"""

MCQ_normal_direct_testing_prompt = """
Answer the MCQ question based on the provided image(s). You need to directly answer question.\n
Output the final answer  in the following format ONLY: <answer> A </answer>\n
The answer must be a single letter: A, B, C, or D.
"""

MCQ_mindcube_cogmap_testing_prompt = """
Answer the MCQ question based on the provided image(s).
Before answering, first build a cognitive map of the scene by integrating all provided views into a single coherent spatial representation on a 10x10 grid.
Use the cognitive map as an intermediate reasoning step to infer the final answer.

Output format:
1. First output the cognitive map wrapped in:
<cogmap> ... </cogmap>

2. Then provide your step-by-step reasoning wrapped in:
<think> ... </think>

3. Finally output the final answer EXACTLY in the following format ONLY:
<answer> A </answer>

The final answer must be a single letter: A, B, C, or D.
"""

MCQ_mindcube_cogmap_direct_testing_prompt = """
Answer the MCQ question based on the provided image(s).
Before giving the final answer, first build a cognitive map of the scene by integrating all provided views into a single coherent spatial representation on a 10x10 grid.

Output format:
1. First output the cognitive map wrapped in:
<cogmap> ... </cogmap>

2. Finally output the final answer EXACTLY in the following format ONLY:
<answer> A </answer>

The final answer must be a single letter: A, B, C, or D.
"""


counting_normal_testing_prompt ="""
Answer the counting question based on the provided image(s). You may reason step by step before answering.\n
After you finish your reasoning, output the final answer in the following format ONLY: <answer> INTEGER </answer>\n
The answer must be a single non-negative integer.
"""

counting_normal_direct_testing_prompt ="""
Answer the counting question based on the provided image(s). You need to directly answer question.\n
Output the final answer in the following format ONLY: <answer> INTEGER </answer>\n
The answer must be a single non-negative integer.
"""

tool_function_mapping = {
    "where": "return_mcq"
}

counting_testing_tool = {
  "type": "function",
  "function": {
    "name": "return_count",
    "description": "Return the counted integer answer.",
    "parameters": {
      "type": "object",
      "properties": {
        "answer": { "type": "integer" }
      },
      "required": ["answer"]
    }
  }
}


MCQ_testing_tool = {
  "type": "function",
  "function": {
    "name": "return_mcq",
    "description": "Return the chosen MCQ option label.",
    "parameters": {
      "type": "object",
      "properties": {
        "answer": { "type": "string" }
      },
      "required": ["answer"]
    }
  }
}

detection_testing_tool = {
  "type": "function",
  "function": {
    "name": "return_bbox",
    "description": "Return a single flat list of bounding boxes, one bounding box per target instance. Each bounding box is in normalized xyxy format [x1, y1, x2, y2].",
    "parameters": {
      "type": "object",
      "properties": {
        "answer": {
          "type": "array",
          "description": "Return a single flat list of bounding boxes. Bounding boxes in normalized xyxy format: each item is [x1, y1, x2, y2].",
          "items": { 
              "type": "array",
              "items": { "type": "number", "minItems": 4, "maxItems": 4 }
          }
        }
      },
      "required": ["answer"]
    }
  }
}
