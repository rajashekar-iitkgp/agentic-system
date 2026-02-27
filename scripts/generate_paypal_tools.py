import json
import os
import re
from typing import List, Dict, Any

COLLECTION_PATH = "app/tools/api_tools/paypal/PayPal APIs.postman_collection.json"
OUTPUT_DIR = "app/tools/api_tools/paypal/generated"

def slugify(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]', '_', text.lower()).strip('_')

def extract_tools_from_item(item: Dict[str, Any], tools: List[Dict[str, Any]]):
    if "item" in item:
        for sub_item in item["item"]:
            extract_tools_from_item(sub_item, tools)
    elif "request" in item:
        req = item["request"]
        name = item["name"]
        method = req["method"]
        url_data = req.get("url", {})
        
        if isinstance(url_data, dict):
            path = "/".join(url_data.get("path", []))
        else:
            path = url_data

        if any(domain in path for domain in ["invoicing", "disputes", "payments"]):
            tools.append({
                "name": slugify(name),
                "description": item.get("description") or f"{method} {path}",
                "method": method,
                "path": path,
                "body": req.get("body", {}),
                "original_name": name
            })

def generate_python_tool(tool: Dict[str, Any]):
    return f"""
# Tool: {tool['original_name']} 
# Path: {tool['method']} /{tool['path']}

class {tool['name'].title().replace('_', '')}Schema(BaseModel):
    # Auto-generation of fields from Postman body would go here
    pass

@async_retry()
async def {tool['name']}(**kwargs):
    \"\"\"{tool['description']}\"\"\"
    # API implementation logic
    pass
"""

def main():
    if not os.path.exists(COLLECTION_PATH):
        print(f"Error: {COLLECTION_PATH} not found.")
        return

    with open(COLLECTION_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tools = []
    extract_tools_from_item(data, tools)
    
    print(f"Extracted {len(tools)} potential tools from Postman.")
    
    for t in tools[:5]:
        print(f"Found Tool: {t['name']} -> {t['method']} /{t['path']}")

if __name__ == "__main__":
    main()
