import os
import json
import base64
import requests
import io
from dotenv import load_dotenv
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# We'll create the OpenAI client only when needed to avoid any initialization issues
def get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def fetch_image_as_base64(image_url: str) -> str:
    """
    Fetch an image from URL and convert it to base64 data URI.
    Returns a data URI string like: data:image/jpeg;base64,/9j/4AAQ...
    """
    try:
        # Add headers to avoid being blocked by CDNs
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # Fetch the image
        response = requests.get(image_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Open and convert to RGB to ensure compatibility
        image = Image.open(io.BytesIO(response.content))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Return as data URI
        return f"data:image/jpeg;base64,{image_base64}"
        
    except Exception as e:
        raise RuntimeError(f"Failed to fetch and convert image: {str(e)}")

def classify_image_screenshot(image_url: str):
    """
    Send an image to gpt-5-nano and classify whether it's a screenshot.
    Fetches the image from URL, converts to base64, and sends to OpenAI.
    Returns a Python dict with keys 'is_screenshot' (bool) and 'reason' (str).
    """
    try:
        # Fetch image and convert to base64
        image_data_uri = fetch_image_as_base64(image_url)

        print("image_data_uri",image_data_uri)
        
        # Create a fresh client each time to avoid any issues with proxies
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-5-nano", # Using available model in OpenAI 1.12.0
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict JSON-only responder. "
                        "Look at the image and decide if it is a screenshot. "
                        "Definition of screenshot:"
                        "- Captured from a phone, tablet, or desktop screen."
                        "- Often shows app UI (navigation bars, buttons, status bars, social media chrome, etc.)."
                        "- May include black, white, or gray bars above or below the main image area."
                        "Return ONLY a JSON object like this:"
                        '{"is_screenshot": true/false, "reason": "<short reason>"}'
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Classify this image."},
                        {"type": "image_url", "image_url": {"url": image_data_uri}}
                    ]
                }
            ],

        )

        print("open ai response",response)
        
        # Extract text output and parse JSON
        output_text = response.choices[0].message.content.strip()
        try:
            result_json = json.loads(output_text)
            return result_json
        except json.JSONDecodeError:
            # In case model fails JSON compliance, wrap in fallback
            return {"is_screenshot": False, "reason": "Could not parse JSON: " + output_text}
    except Exception as e:
        # Fallback in case of any OpenAI API errors
        return {"is_screenshot": False, "reason": f"Error calling OpenAI API: {str(e)}. Assuming not screenshot for safety."}

# Example usage:
if __name__ == "__main__":
    test_url = "https://cdn.discordapp.com/attachments/1093165395466268743/1420669832843497596/image.png?ex=68d63d62&is=68d4ebe2&hm=aa65b63abb273efc987fb32c5951daf44590d774d226d40fcfd2d443c39e74a3&"
    test_url1 = "https://cdn.discordapp.com/attachments/1093165395466268743/1420668169030467584/Screenshot_20250925-130753.png?ex=68d63bd6&is=68d4ea56&hm=e71666bf05808fc3eac1ccf69dffc46fd7a55039f9b877738d2eed7264320dc1&"

    result = classify_image_screenshot(test_url)
    result1 = classify_image_screenshot(test_url1)

    print(result, result1)  # {'is_screenshot': True, 'reason': 'Image shows mobile UI with status bar'}