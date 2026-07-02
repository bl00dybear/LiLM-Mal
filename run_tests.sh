#! /bin/bash

PID_1=231583
PID_2=231584

SPINNER="/-\|"
i=0

while kill -0 $PID_1 2>/dev/null || kill -0 $PID_2 2>/dev/null; do
    printf "\rWaiting for previous script... ${SPINNER:i++%${#SPINNER}:1}"
    sleep 1
done

sleep 30

uv run qwen-1b-lora-classic-ddp/test_elf_set.py

uv run qwen-1b-lora-classic-ddp/test_pe_set.py
