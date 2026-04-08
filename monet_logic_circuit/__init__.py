"""
Monet -> Logic Circuit Conversion

Research project investigating conversion of Monet mixture-of-experts
language model experts into logic circuits for efficient inference.

Pipeline:
    Step 0: Baseline pretrained Monet evaluation
    Step 1: Gentle quantization baseline (8-bit / 4-bit)
    Step 2: Aggressive quantization baseline (2-bit / 1.58-bit ternary)
    Step 3: Logic circuit conversion (exact + learned)
"""

__version__ = "0.1.0"
