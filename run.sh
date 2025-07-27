 #!/bin/bash

# Stop if any command fails
set -e

# Build the Docker image
docker build -t wiki-rag-bot .

# Run the container
docker run --rm -it \
  --env-file .env \
  wiki-rag-bot
