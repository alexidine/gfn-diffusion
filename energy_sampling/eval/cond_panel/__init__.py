"""Offline, per-condition evaluation of conditional crystal checkpoints.

sampler.py rebuilds a checkpoint's policy and reward and draws N samples for a chosen
condition; harness_check.py is the gate that has to pass before any per-molecule number
from it is read: the offline sampler must reproduce the trainer's own logged eval.
"""
