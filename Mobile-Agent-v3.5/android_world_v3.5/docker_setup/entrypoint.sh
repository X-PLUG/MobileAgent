#!/bin/bash

# Start Emulator
#============================================
./docker_setup/start_emu_headless.sh && \
adb root && \
python3 -m server.android_server
