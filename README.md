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

## Run the ADK Developer UI (Desktop/Cloud Shell)

In this dir, start the adk web debug UX:

```bash
cd ads_video_generation_agent
pip install -r requirements.txt
cd ..
adk web
```

## Deploy to Cloud Run

Change 2 parameter in the ads_video_generation_agent/agent.py

```bash
bucket_id = "YOUR-BUCKET-ID"
firestore_database_id = "YOUR-FIRESTORE-DATABASE-ID"
```

Then setup environment variables using command

```bash
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"  # change to your GCP project ID
export GOOGLE_CLOUD_LOCATION="us-central1"  # change to your GCP location
export AGENT_PATH="./ads_video_generation_agent"
export SERVICE_NAME="ads-video-generation-agent-service"
export APP_NAME="ads-video-generation-agent-app"
```

Then deploy to Cloud Run using ADK CLI

```bash
adk deploy cloud_run \
--project=$GOOGLE_CLOUD_PROJECT \
--region=$GOOGLE_CLOUD_LOCATION \
--service_name=$SERVICE_NAME \
--app_name=$APP_NAME \
--with_ui \
$AGENT_PATH
```