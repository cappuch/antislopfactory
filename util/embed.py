import json
import urllib.request

API_URL = "https://infer.dev.takara.ai/"
API_KEY = "EzjkCZ6r1kqhLJGPSsSNmDTGZvZGRbHrmU3cmRkcVlw"


def embed(inputs: str | list[str]) -> list[list[float]]:
    """Get embeddings for one or more texts.

    Args:
        inputs: A single string or list of strings to embed.

    Returns:
        List of embedding vectors (list of floats).
    """
    if isinstance(inputs, str):
        inputs = [inputs]

    payload = json.dumps({"inputs": inputs}).encode()
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": API_KEY,
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


if __name__ == "__main__":
    vectors = embed("Hello world")
    print(f"dims: {len(vectors[0])}")
    print(f"first 5: {vectors[0][:5]}")
