import os
from openai import OpenAI
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client with only the required parameters for version 1.12.0
# No proxies parameter which causes the error
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def classify_image_screenshot(image_url: str):
    """
    Send an image to GPT-5-nano and classify whether it's a screenshot.
    Returns a Python dict with keys 'is_screenshot' (bool) and 'reason' (str).
    """
    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",
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
                        {"type": "image_url", "image_url": image_url}
                    ]
                }
            ],
            max_tokens=300
        )
        
        # Extract text output and parse JSON
        output_text = response.choices[0].message.content.strip()
        try:
            result_json = json.loads(output_text)
            return result_json
        except json.JSONDecodeError:
            # In case model fails JSON compliance, wrap in fallback
            return {"is_screenshot": None, "reason": "Could not parse JSON: " + output_text}
    except Exception as e:
        # Fallback in case of any OpenAI API errors
        return {"is_screenshot": True, "reason": f"Error calling OpenAI API: {str(e)}. Assuming screenshot for safety."}

# Example usage:
if __name__ == "__main__":
    test_url = "https://cdn.discordapp.com/attachments/1093165395466268743/1420669832843497596/image.png?ex=68d63d62&is=68d4ebe2&hm=aa65b63abb273efc987fb32c5951daf44590d774d226d40fcfd2d443c39e74a3&"
    test_url1 = "https://cdn.discordapp.com/attachments/1093165395466268743/1420668169030467584/Screenshot_20250925-130753.png?ex=68d63bd6&is=68d4ea56&hm=e71666bf05808fc3eac1ccf69dffc46fd7a55039f9b877738d2eed7264320dc1&"

    result = classify_image_screenshot(test_url)
    result1 = classify_image_screenshot(test_url1)

    print(result, result1)  # {'is_screenshot': True, 'reason': 'Image shows mobile UI with status bar'}