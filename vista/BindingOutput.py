from pydantic import BaseModel
from typing import Literal

# definisci lo schema con pydantic
class Detection(BaseModel):
    bbox_2d: list[int]   # ideale: tuple di 4
    label: Literal[
        "intact car", "crashed car", "overturned car", "emergency vehicle",
        "person standing", "person walking", "person running",
        "person sitting", "person lying down", "person injured",
        "person helping", "emergency responder"
    ]

class DetectionList(BaseModel):
    detections: list[Detection]

# # costruisci il parser
# parser = JsonSchemaParser(DetectionList.schema())
# prefix_function = build_transformers_prefix_allowed_tokens_fn(self.processor.tokenizer, parser)

# # usalo nella generate
# out_ids = self.model.generate(
#     **inputs,
#     prefix_allowed_tokens_fn=prefix_function,
#     **self.sampling_params,
# )