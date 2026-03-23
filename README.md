# Garment_Prediction

Core Focus: Prediction of how a garment would look when worn on a specific body type (Simple T-shirt, Pants to limit the scope) 

Stage 1: Use IssacSim with PhysX to generate the training data. I plan on using the physics engine to simulate how fabrics drape depending on different body shape parameters (via SMPL), garment size, and material properties (stiffness, density, etc. from fabric descriptions). The dataset generated serves as the ground truth since I am using PhysX. Hope to generate around 1000 different samples. 

Stage 2: Then, train a NN to approximate the physics simulator. Input would be the body shape parameters, garment size, and material properties. Then, the output of this draping model would be the vertex positions of the draped garment mesh and thus skip the physics simulation. Loss is the distance between predicted and simulated vertex positions. 

Stage 3 (Reach Goal):  Simulate how a garment would look from a user-provided image. User scrolls through Gap website, use SAM to segment the garment. Train a CNN model to estimate the necessary material properties for our draping model. Use the metadata on Gap website as supplementary input and the ground truth is the data provided from Stage 1. Model looks at image, predicts what parameters would produce that drape, and apply to mannequin with different body shape parameters. Loss would be MSE between predicted material parameters and known material parameters. 
