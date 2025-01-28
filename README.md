# esp-data
This repository contains data tools for managing data operations in ESP's AI / science projects.

## Setup
1. Are you (or a service account for a VM / cloud service) authenticated with gcp ?
    On a local development environment, make sure *gcloud* is installed. Then either:
    > * Run ```gcloud init```, select the project and sign-in or,
    > * If gcloud is already configured to the right project, run ```gcloud auth login``` and ```gloud auth application-default login```, which will use your google earthspecies account for authentication or,
    > * If you are on a gcp VM using the default service account / VM service account or workload indentity verification.

2. Clone this repo
3. cd into root
4. Run:
   >
   > ```uv sync```
5. Run tests: In root, run ```uv run pytest tests```. 

If tests pass, you're set up!
