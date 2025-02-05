# esp-data
This repository contains data tools for managing data operations in ESP's AI / science projects.

## Setup
1. Are you (or a service account for a VM / cloud service) authenticated with gcp ?
    On a local development environment, make sure *gcloud* is installed. Then either:
    > * Run ```gcloud init```, select the project and sign-in or,
    > * If gcloud is already configured to the right project, run ```gcloud auth login``` and ```gloud auth application-default login```, which will use your google earthspecies account for authentication or,
    > * If you are on a gcp VM using the default service account / VM service account or workload indentity verification.

2. Install uv. ```curl -LsSf https://astral.sh/uv/install.sh | sh```
3. Clone this repo
4. cd into root
5. Run ```uv sync```
6. Install dev version ```uv pip install -e .``` (TODO: This is a temporary fix so that pythonpath recognises esp_data package)
7. Run tests: In root, run ```uv run pytest tests```.

If tests pass, you're set up!
