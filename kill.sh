#!/bin/bash
# kills gz, which is run by ruby. This will kill other ruby processes.
echo "killing all ruby processes"
pkill -9 ruby