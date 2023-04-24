#! /bin/bash

# For loop for the number of agents

for i in {5..50..5}
do
    # Save the image
    python save_images.py --num_agent "$i"
done