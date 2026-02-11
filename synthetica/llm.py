from openai import OpenAI

class OpenAIClient:
    def __init__(self, api_key: str | None = None):
        self.client = OpenAI(api_key=api_key)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content