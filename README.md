# Ads Creative Generation Agent

This project contains an agent to generate creative ads videos based on the user's requirement.

## Prerequisites

Install the MCP Servers for Genmedia tools. This example uses the Go versions and assumes you've installed them locally.

## Setup

Add an .env file to the `ads_video_generation_agent` agent directory:

```bash
GOOGLE_CLOUD_PROJECT="your-project-id"
GOOGLE_CLOUD_LOCATION="your-location" #e.g. us-central1
GOOGLE_GENAI_USE_VERTEXAI="True"
GCS_BUCKET_NAME="your-gcs-bucket-name"
FIRESTORE_DATABASE_ID="your-firestore-database-id"
```

## Run the ADK Developer UI

In this dir, start the adk web debug UX:

```bash
uv sync
source .venv/bin/activate
adk web
```
