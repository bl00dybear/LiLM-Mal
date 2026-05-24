#! /bin/bash

uv run qwen-3b-unfroze-fsdp/test_elf_set.py

uv run qwen-3b-lora-classic-ddp/test_elf_set.py
