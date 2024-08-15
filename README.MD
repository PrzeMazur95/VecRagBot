
# Vector Search App with Qdrant

This project demonstrates a simple vector search application using Qdrant to query through six statically vectorized cities. Follow the instructions below to set up and run the application.

## Prerequisites

Make sure you have the following installed:

- Docker
- Docker Compose

## Overview

After running the commands below, you'll be able to see the closest stored city in Qdrant to a provided query vector in your terminal.

## Setup Instructions

### Step 1: Clean Up Existing Docker Containers and Images

If you've previously set up the project and want to start fresh, you'll need to remove any existing Docker containers and images.

1. **Find Container IDs:**
   ```sh
   docker container ps -a
   ```

2. **Stop and Remove Containers:**
   ```sh
   docker stop vecragbot qdrant && docker rm vecragbot qdrant
   ```

3. **Remove the Python App Image:**
   ```sh
   docker rmi vecragbot-pythonapp
   ```

### Step 2: Start the Application

1. **Build and Start the Containers:**
   ```sh
   docker compose up --build -d
   ```

   - This command will rebuild images if they already exist, ensuring the containers run on the correct images.
   - If no previous images exist, new ones will be created.

### Step 3: Run the Vector Search

1. **Access the `vecragbot` Container:**
   Open a new terminal window and run:
   ```sh
   docker exec -it vecragbot sh
   ```

2. **Execute the Python Script:**
   In the container terminal, run:
   ```sh
   python3 ./main.py
   ```

   - This script installs all necessary data, creates a collection in Qdrant, feeds it with six statically vectorized cities, and returns the nearest point from the provided vector in `main.py`.

## Conclusion

You should now have a fully functioning vector search application running on Qdrant, capable of querying and finding the closest city to a provided vector. Enjoy exploring the capabilities of Qdrant with this example!

![app_visualisation](.doc/app_visualisation.png)