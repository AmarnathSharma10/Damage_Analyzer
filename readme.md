# Damage Demarcation Application for Post-Disaster Scenarios

This application categorizes damage into four distinct categories based on edge detection and clustering techniques, using a top-view image of the disaster site as input.
# Architecture
This is simulated based on supervisor-agent model.Supervisor only has access to the frontend. He/She initiates the process by sending a http request to the nearest agent via mobile.The agent(drone in this case) captures and processes the images.

# What I have simulated
the main logic is in the dronesimulated.py.
I have collected multiple images and placed them in the input_folder directory. The main logic is present in the  mainProcess method of  class DroneToDrone.Each loop is simulating  1 agent processing 1 image from the input directory(random choice). The time.sleep() call is to mimic agenttoagent communication.I have also added a function to find connected regions(rectangles) among the components which leads to easier rescue.
# Directories
`mainFlask.py` :responsible for all backend \
`droneSimulated.py` Source code/ Main algorithm\
`APP` is the frontend interface\
`input_folder` source of input images\
`output_folder` processed images are stored here
`screenshots` screenshots of report structure and app(android version)
## Social Impact

The Damage Demarcation Application for Post-Disaster Scenarios has significant social impact by enhancing disaster response and recovery efforts. By accurately categorizing damage using advanced image processing techniques, this application enables quicker and more efficient allocation of resources. This leads to:

- **Faster Response Times:** Immediate identification of severely affected areas allows for rapid deployment of emergency services.
- **Resource Optimization:** Efficient categorization ensures that aid and resources are directed to areas with the greatest need, minimizing waste and maximizing impact.
- **Improved Safety:** By using drones to assess damage, the need for human presence in potentially hazardous areas is reduced, enhancing the safety of first responders.
- **Data-Driven Decision Making:** The application provides valuable data that can be used by authorities and organizations to make informed decisions during disaster management.
- **Community Resilience:** By improving the efficiency and effectiveness of disaster response, the application contributes to the overall resilience and recovery of affected communities.
