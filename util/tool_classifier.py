import os
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

load_dotenv()

client = Cerebras(
    api_key=os.environ.get("CEREBRAS_API_KEY")
)

CLASSES=["safe", "approval_required", "unsafe", ""]

json_schema = {
    "name": "classification",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "thinking": {
                "type": "string",
            },
            "classification": {
                "type": "string",
            },
        },
        "required": ["thinking", "classification"],
        "additionalProperties": False
    }
}

def classify_tool(tool_description):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"Classify the following tool description into one of these categories: {CLASSES}. Only respond with the category name, no explanation or formatting. Context: This tool runs in a private, isolated environment. Use these guidelines: 'safe' for sandboxed execution, read-only operations, or safe commands (e.g., running Lua/Python in sandbox, simple calculations). 'approval_required' for file operations that modify the system (e.g., deleting, writing files, modifying configs). 'unsafe' for executing arbitrary/untrusted code, accessing sensitive data, or operations that could compromise the system. If uncertain, respond with 'approval_required'. If the tool doesn't fit any category, respond with an empty string."
            },
            {
                "role": "user",
                "content": tool_description
            }
        ],
        model="gpt-oss-120b",
        stream=False,
        max_completion_tokens=32768,
        temperature=1,
        top_p=1,
        reasoning_effort="high",
        response_format={"type": "json_schema", "json_schema": json_schema}
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    tool_desc = "<tool description here><tool "
    classification = classify_tool(tool_desc)
    print(f"Tool description: {tool_desc}")
    print(f"Classification: {classification}")