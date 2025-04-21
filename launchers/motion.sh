#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init


# launch subscriber
rosrun my_package motion.py 3

# wait for app to end
dt-launchfile-join

# python3 packages/my_package/src/shutdown.py