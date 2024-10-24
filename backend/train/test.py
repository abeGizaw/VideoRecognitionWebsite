import fiftyone as fo
import fiftyone.zoo as foz

#
# Load 10 random samples from the validation split
#
# Only the required videos will be downloaded (if necessary).
#

import fiftyone.zoo as foz

# dataset = foz.load_zoo_dataset("kinetics-700-2020")

dataset = foz.load_zoo_dataset(
    "kinetics-700-2020",
    split="validation",
    max_samples=10,
    shuffle=True,
)

session = fo.launch_app(dataset)
