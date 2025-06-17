#!/bin/bash

uv run baseline.py \
--repr_time_prey 500 \
--prey_amount 100 \
--predator_amount 4 \
--death_time_predator 300 \
--repr_amount_predator 10 \
--eat_probability 0.01 \
--duration 100 \
--frame_limit 0
