#!/bin/bash

ps axu | grep translate1.py | awk '{print $2}' | xargs kill -9
