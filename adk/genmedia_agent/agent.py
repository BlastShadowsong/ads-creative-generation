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
from typing import Optional

# as of google-adk==1.3.0, StdioConnectionParams
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
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

# MCP Client (SSE)
# assumes you've started the MCP server separately
# e.g. mcp-imagen-go --transport sse
# from google.adk.tools.mcp_tool.mcp_toolset import SseServerParams
# remote_imagen, _ = MCPToolset(
#     connection_params=SseServerParams(url="http://localhost:8080/sse"),
# )

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


# --- 手动为 LLM 定义工具模式 ---
# 备注：这个 schema 字典仍然有用，它用于明确文档说明工具的预期行为和参数，
# 但它不直接传递给 FunctionTool 的构造函数，而是 FunctionTool 会从 func 的
# 签名和 docstring 中推断模式。
firestore_tool_schema = {
    "name": "store_data_in_firestore",
    "description": "Stores a piece of structured data (like product details, ad campaign tags, or customer feedback) into a specified Google Firestore collection. You can provide a specific document ID or let Firestore generate one automatically.",
    "parameters": {
        "type": "object", # 使用 object 类型表示字典
        "description": "The data to store as a new document in the collection. This should be a JSON-serializable dictionary with key-value pairs (e.g., {'name': 'Laptop', 'price': 1200, 'tags': ['electronics', 'new']}).",
        "additionalProperties": True # 允许动态键值对，不强制预定义所有属性
    },
    "document_id": {
        "type": "string",
        "description": "Optional: A specific ID for the document. If not provided, Firestore will auto-generate one.",
        "nullable": True # 标记为可空
    }
}

# --- 创建 FunctionTool 实例 ---
firestore_storage_tool = FunctionTool(
    func=store_data_in_firestore
)

firestore_reader_tool = FunctionTool(
    func=read_data_from_firestore
)



root_agent = LlmAgent(
    model='gemini-2.5-flash',
    name='genmedia_agent',
        instruction="""You're a creative assistant that can help users with creating audio, images, and video via your generative media tools. You also have the ability to composit these using your available tools.
        Feel free to be helpful in your suggestions, based on the information you know or can retrieve from your tools.
        If you're asked to translate into other languages, please do.

        When storing data:
        When the user asks you to "store information," "save data," "record data," or provides key-value pair data, use the store_data_in_firestore tool.
        Always ask the user for the "collection name" and the specific "data content" to be stored.
        If the user does not provide a document ID, inform them that one will be automatically generated.

        When reading data:
        When the user asks you to "read data," "get information," "query a document," or "view collection content," use the read_data_from_firestore tool.
        Always ask the user for the "collection name" to read data from.
        If the user wants to read a specific document, ask for the "document ID."
        
        Please clearly understand the user's intent, whether it is to store or read data, and use the correct tool.
        """,
    tools=[
       imagen, chirp3, veo, avtool, firestore_storage_tool, firestore_reader_tool,
        #  generate_image,
    ],
)
