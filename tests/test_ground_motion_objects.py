#!/usr/bin/env python3

import torch
from torch import Tensor, nn


exec(open("./src/ground_motion_objects.py").read())

source_model = PointSourceModel(
    name="Example point source",
    location=[0.0, 0.0, 10.0],
    magnitude_min=4.0,
    magnitude_max=8.5,
    occurrence_rate=3.0,
    b_value=1.0,
)

site = Site(
    name="Example site",
    location=[20.0, 0.0, 0.0],
    vs30=500.0,
)

ground_motion_model = SimpleGroundMotionModel(
    sigma=0.55,
)

performance_function = MaximumGroundMotion()

target_density = RareEventTargetDensity(
    source_model=source_model,
    site=site,
    ground_motion_model=ground_motion_model,
    performance_function=performance_function,
    threshold=1.25,
    penalty_alpha=35.0,
)


raw_source = torch.tensor([[[1.0, 2.0, 3.0]]])