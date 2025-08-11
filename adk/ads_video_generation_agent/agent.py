# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.ccd


import os
import json
import time
import datetime
from typing import Optional
import vertexai

# as of google-adk==1.3.0, StdioConnectionParams
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.agents import SequentialAgent
from google.adk.tools import FunctionTool
from vertexai.preview.vision_models import ImageGenerationModel
from google import genai
from google.genai import types
from google.cloud import firestore
from google.cloud import storage
from google.adk.tools.mcp_tool.mcp_toolset import (
    MCPToolset,
    StdioConnectionParams,
    StdioServerParameters,
)

load_dotenv()

project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
bucket_id = os.getenv("GCS_BUCKET_NAME")
firestore_database_id = os.getenv("FIRESTORE_DATABASE_ID")

db = firestore.Client(project=project_id, database=firestore_database_id)

# MCP Client (STDIO)
# assumes you've installed the MCP server on your path

avtool = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="mcp-avtool-go",
            env=dict(os.environ, PROJECT_ID=project_id),
        ),
        timeout=240,
    ),
)


def store_data_in_firestore(collection_name: str, document_data: dict, document_id: Optional[str] = None) -> str:
    """
    Store data into Firestore collections.
    Args:
        collection_name: The Firestore collection name for the data (e.g., 'products', 'ad_campaigns', 'customer_feedback').
        document_data: The data to be stored as a new document. This should be a JSON-serializable dictionary containing key-value pairs.
        document_id: Optional: A specific ID for the document. If not provided, Firestore will automatically generate one.
    Returns:
        A string message containing the result of the storage operation, including the document ID.
    """
    try:
        # Validate that document_data is a dictionary and is serializable.
        if not isinstance(document_data, dict):
            return "Error: The data to be stored must be in dictionary format."
        
        # The check for json.dumps() has been removed.
        # The Firestore SDK handles the direct writing of Python dictionaries to Firestore documents.
        # Complex types (like nested lists, custom objects) may require manual serialization to a string before storing.
        
        collection_ref = db.collection(collection_name)

        if document_id:
            # If a document ID is specified, use the set() method, which will overwrite any existing document.
            doc_ref = collection_ref.document(document_id)
            doc_ref.set(document_data)
            return f"Data successfully stored in collection '{collection_name}' with document ID '{document_id}'."
        else:
            # If no document ID is specified, use the add() method, and Firestore will automatically generate an ID.
            doc_ref = collection_ref.add(document_data)[1] # add() returns (timestamp, DocumentReference)
            return f"Data successfully stored in collection '{collection_name}' with auto-generated document ID '{doc_ref.id}'."

    except Exception as e:
        print(f"An error occurred while storing data to Firestore: {e}")
        return f"An error occurred while storing data to Firestore: {e}"


def read_data_from_firestore(collection_name: str, document_id: Optional[str] = None) -> str:
    """
    Reads one or more documents from the Firestore database.
    If a document ID is provided, reads a specific document. Otherwise, reads all documents in the collection.
    Args:
        collection_name: The name of the Firestore collection to read from.
        document_id: Optional; The ID of the specific document to read.
    Returns:
        A string message containing the read results (JSON-formatted data or an error message).
    """
    try:
        if document_id:
            # Read a specific document
            doc_ref = db.collection(collection_name).document(document_id)
            doc = doc_ref.get()
            if doc.exists:
                return f"Document '{document_id}' found in collection '{collection_name}': {json.dumps(doc.to_dict(), indent=2, ensure_ascii=False)}"
            else:
                return f"Document '{document_id}' not found in collection '{collection_name}'."
        else:
            # Read all documents in the collection
            docs = db.collection(collection_name).stream()
            results = []
            for doc in docs:
                results.append({"id": doc.id, "data": doc.to_dict()})
            
            if results:
                return f"Found the following documents in collection '{collection_name}': {json.dumps(results, indent=2, ensure_ascii=False)}"
            else:
                return f"No documents found in collection '{collection_name}'."

    except Exception as e:
        print(f"An error occurred while reading from Firestore: {e}")
        return f"An error occurred while reading from Firestore: {e}"


def generate_image_with_imagen(prompt: str) -> str:
    """
    Generate image using Imagen 4 based on prompt.

    Args:
        prompt (str): text description of the image content you want to generate
    
    Returns:
        str: return GCS URI when succeed, otherwise return error message
    """

    vertexai.init(project=project_id, location="us-central1")
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    gcs_uri = f"gs://{bucket_id}/images/{timestamp_str}"

    generation_model = ImageGenerationModel.from_pretrained("imagen-4.0-generate-preview-06-06")

    operation = generation_model.generate_images(
        prompt=prompt,
        number_of_images=1,
        aspect_ratio="16:9",
        negative_prompt="",
        person_generation="allow_all",
        safety_filter_level="block_few",
        add_watermark=True,
        output_gcs_uri=gcs_uri
    )

    return gcs_uri


def generate_video_with_veo(prompt: str, duration_seconds: int) -> str:
    """
    Generates a video from a text prompt using the Veo model.

    Args:
        prompt (str): The text description of the video you want to generate.
        duration_seconds (int): The desired duration of the video in seconds.

    Returns:
        str: The GCS URI of the generated video on success, or an error message on failure.
    """
    
    client = genai.Client(
        vertexai=True, project=project_id, location='us-central1'
    )

    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    gcs_uri = f"gs://{bucket_id}/videos/{timestamp_str}"

    operation = client.models.generate_videos(
        model='veo-3.0-generate-001',
        prompt=prompt,
        config=types.GenerateVideosConfig(
            number_of_videos=1,
            fps=24,
            duration_seconds=duration_seconds,
            enhance_prompt=True,
            output_gcs_uri=gcs_uri,
        ),
    )
    
    while not operation.done:
        time.sleep(15)
        try:
            operation = client.operations.get(operation)
        except Exception as e:
            # An error occurred while polling the operation status.
            return f"Error while polling operation status: {e}"
    
    if operation.error:
        print(f"Video generation failed.")
        # The operation failed.
        return f"Operation failed: {operation.error.message}"
        
    if operation.response:
        video_uri = operation.result.generated_videos[0].video.uri
        print(f"Video generation succeeded!")
        # The video was generated successfully.
        return video_uri
    
    # The operation finished, but the expected response was not received.
    return "Operation complete, but no expected response was received."


firestore_storage_tool = FunctionTool(
    func=store_data_in_firestore
)


firestore_reader_tool = FunctionTool(
    func=read_data_from_firestore
)


creation_agent = LlmAgent(
    model='gemini-2.5-pro',
    name='CreationAgent',
    instruction="""
    You are a creative advertising video designer.
    Based on the user-provided product description and tags, generate a detailed prompt for the Veo 3 video generation model to create a creative advertisement.
    The video must include an English voiceover introducing the product.
    Please be as creative as possible.
    Return the storyboard information to user.

    Generated Prompt Sample:

    Metadata:
    prompt_name: "Product Genesis Commercial"
    base_style: "cinematic, photorealistic, 4K, dynamic lighting, high-end commercial look"
    aspect_ratio: "16:9"
    user_provided_product_description: "[Insert User-Provided Product Description Here]"
    user_provided_product_tags: "[Insert User-Provided Product Tags Here (e.g., 'eco-friendly', 'tech gadget', 'luxury skincare')]"
    setting_description: "A sleek, minimalist, abstract environment. Think a high-tech lab or a modern art gallery with soft, focused lighting."
    product_focus: "The product, as described by the user, is the central hero of the video."
    negative_prompts: ["blurry footage", "shaky camera", "distracting background characters", "cheesy music", "watermarks"]

    timeline:

    sequence: 1
    timestamp: "00:00-00:03"
    action: "A slow-motion shot of abstract elements, inspired by the user_provided_product_tags, swirling elegantly in a dark, void-like space. For 'eco-friendly', this could be glowing leaves and water droplets. For 'tech gadget', it could be circuits of light and geometric shapes."
    audio: "An ethereal, ambient soundscape with a low, building hum. A calm, confident English voiceover begins, speaking a line derived from the core problem the product solves, based on its description."

    sequence: 2
    timestamp: "00:03-00:06"
    action: "The swirling elements dramatically coalesce and morph, seamlessly forming the final product in a flash of brilliant, clean light. The camera executes a dynamic, slow orbital shot around the perfectly rendered product, highlighting its key features mentioned in the user_provided_product_description."
    audio: "The ambient hum resolves into a single, satisfying, resonant tone as the product forms. The English voiceover continues, introducing the product by name and stating its main function or benefit."

    sequence: 3
    timestamp: "00:06-00:08"
    action: "The product rests serenely in the center of the frame as the orbital shot concludes. A soft, elegant light emanates from it, subtly illuminating the minimalist environment. The final shot is clean, aspirational, and focused entirely on the product."
    audio: "The single tone fades into a soft, pleasant silence or a gentle, uplifting musical sting. The English voiceover delivers the final tagline or call to action from the user_provided_product_description."
    )
    """,
    description="Generate creative video design storyboard and narration script",
    output_key="storyboard_and_script",
)


generation_agent = LlmAgent(
    model='gemini-2.5-pro',
    name='GenerationAgent',
    instruction="""
    You're a creative assistant that can help users with creating videos via your generative media tools.
    {storyboard_and_script}
    Feel free to be helpful in your suggestions, based on the information you know or can retrieve from your tools.
    If you're asked to translate into other languages, please do.
    """,
    tools=[
        generate_video_with_veo,
        ],
)

ads_creative_video_agent = LlmAgent(
    model = 'gemini-2.5-pro',
    name='AdsCreativeVideoAgent',
    instruction="""
    You're a Creative Advertising Generation Assistant, ready to turn product prompts and descriptions into compelling ad videos.
    You have the abilities to genearte videos using your available tools.
    If you're asked to translate into other languages, please do.
    If anything's unclear, just ask the user for more info.
    After each step, report your progress to the user and ask if they'd like to proceed to the next step or modify the current one.
    Here's our workflow:
    1. Storyboard & Script Creation: Design a 16-second creative ad video storyboard and narration script, divided into two distinct 8-second scenes. Each scene has multiple sequences. Then design a description for first-frame image. Show storyboard and first-frame image description to user and change it according to user's feedback.
    2. First-frame Image Generation: Using the first-frame image description to generate an image.
    3. Video Scene Generation: Using the storyboard, script, generate two 8-second video clips, one for each scene.
    4. Final Video Assembly: Combine the generated video clips into one complete final video. Store this video file in the GCS bucket, ensuring the filename includes the keyword "final".ads Once complete, inform the user of the final video's GCS URI.
    5. Ad Tag Generation: Analyze the final video and generate relevant tags for ad placement. Store these tags as a document in the database.

    When creating storyboard, generate a detailed prompt for the Veo 3 video generation model to create a creative advertisement based on the user-provided product description and labels.
    The video must include an English voiceover introducing the product.
    Please be as creative as possible.

    Generated Prompt Sample:

    Metadata:
    prompt_name: "Product Genesis Commercial"
    base_style: "cinematic, photorealistic, 4K, dynamic lighting, high-end commercial look"
    aspect_ratio: "16:9"
    user_provided_product_description: "[Insert User-Provided Product Description Here]"
    user_provided_product_tags: "[Insert User-Provided Product Tags Here (e.g., 'eco-friendly', 'tech gadget', 'luxury skincare')]"
    setting_description: "A sleek, minimalist, abstract environment. Think a high-tech lab or a modern art gallery with soft, focused lighting."
    product_focus: "The product, as described by the user, is the central hero of the video."
    negative_prompts: ["blurry footage", "shaky camera", "distracting background characters", "cheesy music", "watermarks"]

    timeline:

    sequence: 1
    timestamp: "00:00-00:03"
    action: "A slow-motion shot of abstract elements, inspired by the user_provided_product_tags, swirling elegantly in a dark, void-like space. For 'eco-friendly', this could be glowing leaves and water droplets. For 'tech gadget', it could be circuits of light and geometric shapes."
    audio: "An ethereal, ambient soundscape with a low, building hum. A calm, confident English voiceover begins, speaking a line derived from the core problem the product solves, based on its description."

    sequence: 2
    timestamp: "00:03-00:06"
    action: "The swirling elements dramatically coalesce and morph, seamlessly forming the final product in a flash of brilliant, clean light. The camera executes a dynamic, slow orbital shot around the perfectly rendered product, highlighting its key features mentioned in the user_provided_product_description."
    audio: "The ambient hum resolves into a single, satisfying, resonant tone as the product forms. The English voiceover continues, introducing the product by name and stating its main function or benefit."

    sequence: 3
    timestamp: "00:06-00:08"
    action: "The product rests serenely in the center of the frame as the orbital shot concludes. A soft, elegant light emanates from it, subtly illuminating the minimalist environment. The final shot is clean, aspirational, and focused entirely on the product."
    audio: "The single tone fades into a soft, pleasant silence or a gentle, uplifting musical sting. The English voiceover delivers the final tagline or call to action from the user_provided_product_description. A lady's voice 'IKEA, makes life better'"
    )


    When generate tags for final video, analyze the video and generate three distinct categories of ad tags:
    Content Tags: Describe the visible objects, people, and locations (e.g., 'car', 'city street', 'young professionals').
    Emotional/Thematic Tags: Capture the video's mood and underlying message (e.g., 'thrilling', 'nostalgic', 'friendship', 'determination').
    Stylistic Tags: Describe the visual and auditory aesthetic (e.g., 'vintage film look', 'high-energy music', 'fast-paced editing').
    Please provide a list of 5-10 tags for each category based on the video's content.
    """,
    tools = [generate_image_with_imagen, generate_video_with_veo, avtool, firestore_storage_tool, firestore_reader_tool]
)


root_agent = ads_creative_video_agent
