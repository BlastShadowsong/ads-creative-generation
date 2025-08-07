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
# limitations under the License.


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

# --- 初始化 Firestore 客户端 ---
# Firestore 客户端将使用与 Cloud Run 服务关联的服务账户进行认证。
# 确保你的服务账户拥有访问 Firestore 的权限。
db = firestore.Client(project=project_id, database=firestore_database_id)

# MCP Client (STDIO)
# assumes you've installed the MCP server on your path
veo = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="mcp-veo-go",
            env=dict(os.environ, PROJECT_ID=project_id),
        ),
        timeout=60,
    ),
)

chirp3 = MCPToolset(
    connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="mcp-chirp3-go",
                env=dict(os.environ, PROJECT_ID=project_id),
            ),
            timeout=60,
    ),
)

imagen = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="mcp-imagen-go",
            env=dict(os.environ, PROJECT_ID=project_id),
        ),
        timeout=60,
    ),
)

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
    将结构化数据存储到指定的 Firestore 集合中。
    Args:
        collection_name: 数据的 Firestore 集合名称（例如，'products', 'ad_campaigns', 'customer_feedback'）。
        document_data: 要存储为新文档的数据。这应该是一个 JSON 可序列化的字典，包含键值对。
        document_id: 可选：文档的特定 ID。如果未提供，Firestore 将自动生成一个。
    Returns：
        包含存储操作结果的字符串消息，包括文档ID。
    """
    try:
        # 验证 document_data 是否是字典且可序列化
        if not isinstance(document_data, dict):
            return "错误：要存储的数据必须是字典格式。"
        
        # 移除了 json.dumps() 检查。
        # Firestore SDK 会直接处理 Python 字典到 Firestore 文档的写入。
        # 复杂类型（如嵌套列表、自定义对象）可能需要手动序列化为字符串再存储。
        
        collection_ref = db.collection(collection_name)

        if document_id:
            # 如果指定了文档ID，则使用 set() 方法，会覆盖现有文档
            doc_ref = collection_ref.document(document_id)
            doc_ref.set(document_data)
            return f"数据已成功存储在集合 '{collection_name}' 中，文档 ID 为 '{document_id}'。"
        else:
            # 如果未指定文档ID，则使用 add() 方法，Firestore 会自动生成一个ID
            doc_ref = collection_ref.add(document_data)[1] # add() 返回 (timestamp, DocumentReference)
            return f"数据已成功存储在集合 '{collection_name}' 中，自动生成的文档 ID 为 '{doc_ref.id}'。"

    except Exception as e:
        print(f"存储数据到 Firestore 过程中发生错误: {e}")
        return f"存储数据到 Firestore 过程中发生错误: {e}"

# --- 定义 Firestore 读取工具函数 ---
def read_data_from_firestore(collection_name: str, document_id: Optional[str] = None) -> str:
    """
    从 Firestore 数据库中读取一个或多个文档。
    如果提供了文档ID，则读取特定文档。否则，读取集合中的所有文档。
    Args:
        collection_name: 要读取数据的 Firestore 集合名称。
        document_id: 可选：要读取的特定文档的ID。
    Returns:
        包含读取结果的字符串消息（JSON格式的数据或错误消息）。
    """
    try:
        if document_id:
            # 读取特定文档
            doc_ref = db.collection(collection_name).document(document_id)
            doc = doc_ref.get()
            if doc.exists:
                return f"在集合 '{collection_name}' 中找到文档 '{document_id}': {json.dumps(doc.to_dict(), indent=2, ensure_ascii=False)}"
            else:
                return f"在集合 '{collection_name}' 中未找到文档 '{document_id}'。"
        else:
            # 读取集合中的所有文档
            docs = db.collection(collection_name).stream()
            results = []
            for doc in docs:
                results.append({"id": doc.id, "data": doc.to_dict()})
            
            if results:
                return f"在集合 '{collection_name}' 中找到以下文档: {json.dumps(results, indent=2, ensure_ascii=False)}"
            else:
                return f"集合 '{collection_name}' 中没有找到任何文档。"

    except Exception as e:
        print(f"从 Firestore 读取数据过程中发生错误: {e}")
        return f"从 Firestore 读取数据过程中发生错误: {e}"


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
    使用 Veo 模型根据文本提示生成视频。

    Args:
        prompt (str): 描述你想要生成的视频内容的文本。
        duration_seconds (int): 期望的视频时长（秒）。

    Returns:
        str: 成功时返回生成视频的 GCS URI，失败时返回错误信息。
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
            return f"轮询操作状态时出错: {e}"
    
    if operation.error:
        print(f"❌ 视频生成失败。")
        return f"操作失败: {operation.error.message}"
        
    if operation.response:
        video_uri = operation.result.generated_videos[0].video.uri
        print(f"🎉 视频生成成功！")
        return video_uri
    
    return "❌ 操作完成，但未收到预期的响应。"


# --- 创建 FunctionTool 实例 ---
firestore_storage_tool = FunctionTool(
    func=store_data_in_firestore
)

firestore_reader_tool = FunctionTool(
    func=read_data_from_firestore
)


workflow_agent = LlmAgent(
    model='gemini-2.5-pro',
    name='genmedia_agent',
    instruction="""
    You're a Creative Advertising Generation Assistant, ready to turn product images and descriptions into compelling ad videos. 
    You have the abilities to composit images, audios, videos using your available tools.
    If you're asked to translate into other languages, please do.
    If anything's unclear, just ask the user for more info.
    Important: Don't return any generated assets directly. Instead, store all results in the gs://sample-ads-creative GCS bucket. Name files using the format: [content_type]_[timestamp] (e.g., image_1703408000.png, video_1703408000.mp4).
    After each step, report your progress to the user and ask if they'd like to proceed to the next step or modify the current one.
    Here's our workflow:
    1. Storyboard & Script Design: Design a 32-second creative ad video storyboard and narration script, divided into four distinct 8-second scenes.
    2. Scene Keyframe Generation: Based on the designed storyboard and script, generate one keyframe image for each of the four scenes. Store these image files in the GCS bucket.
    3. Video Scene Generation: Using the storyboard, script, and keyframe images, generate four 8-second video clips, one for each scene. Store these video files in the GCS bucket.
    4. Narration Voice-over Production: Based on the script, produce narration voice-over audio for each scene. Store these audio files in the GCS bucket.
    5. Final Video Assembly: Combine the generated video clips and narration voice-overs into one complete final video. Store this video file in the GCS bucket, ensuring the filename includes the keyword "final". Once complete, inform the user of the final video's GCS URI.
    6. Ad Tag Generation: Analyze the final video and generate relevant tags for ad placement. Store these tags as a document in the database.
    """,
    tools=[
       imagen, chirp3, veo, avtool, firestore_storage_tool, firestore_reader_tool,
        #  generate_image,
    ],
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
    4. Final Video Assembly: Combine the generated video clips and narration voice-overs into one complete final video. Store this video file in the GCS bucket, ensuring the filename includes the keyword "final".ads Once complete, inform the user of the final video's GCS URI.
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

# ads_creative_video_pipeline_agent = SequentialAgent(
#     name='AdsCreativeVideoPipelineAgent',
#     sub_agents=[creation_agent, generation_agent],
#     description="Executes a sequence of video storyboard creation, video generation, and labeling.",
# )

root_agent = ads_creative_video_agent
