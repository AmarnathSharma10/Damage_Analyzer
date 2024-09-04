# Damage Demarcation Application for Post-Disaster Scenarios

This application categorizes damage into four distinct categories based on edge detection and clustering techniques, using a top-view image of the disaster site as input.
# Architecture
This is simulated based on supervisor-agent model.Supervisor only has access to the frontend. He/She initiates the process by sending a http request to the nearest agent via mobile.The agent(drone in this case) captures and processes the images.

# What I have simulated
the main logic is in the dronesimulated.py.
I have collected multiple images and placed them in the input_folder directory. The main logic is present in the  mainProcess method of  class DroneToDrone.Each loop is simulating  1 agent processing 1 image from the input directory. The time.sleep() call is to mimic agenttoagent communication
