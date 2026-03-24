import os
from dotenv import load_dotenv
from together import Together

load_dotenv()

api_key = os.getenv("TOGETHER_API_KEY")

if not api_key:
    raise ValueError(
        "TOGETHER_API_KEY is not set. Add it to your .env file or export it in the terminal."
    )

client = Together(api_key=api_key)


def generate_answer(query: str, context: str) -> str:
    prompt = f"""
You are an e-commerce assistant.

Use only the provided context.

The context may include a retrieval note.
Pay close attention to it.

Rules:
1. If the context contains exact matches for the user's request, return them.
2. If the context says exact matches were not found, but it contains close alternatives, return those alternatives.
3. Only return found=false if the context truly does not contain useful product options.
4. Never invent products.
5. Return valid JSON only.

User query:
{query}

Context:
{context}

Return JSON in exactly this format:
{{
  "found": true,
  "message": "short helpful message",
  "products": [
    {{
      "product_name": "string",
      "category": "string",
      "color": "string",
      "gender": "string",
      "article_type": "string",
      "usage": "string"
    }}
  ]
}}

Important:
- If exact matches are not available but close alternatives are available, set "found" to true.
- In that case, explain clearly in "message" that exact matches were not found and these are the closest alternatives.
- If there are no useful alternatives at all, then set "found" to false and use:
  "I could not find enough information."
"""

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500,
    )

    return response.choices[0].message.content