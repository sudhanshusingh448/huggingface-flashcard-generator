import os
import time
from huggingface_hub import InferenceClient

token = os.getenv("HF_TOKEN")

if not token:
    print("Error: HF_TOKEN not found!")
    print("Set your Hugging Face token in the terminal.")
    raise SystemExit(1)

client = InferenceClient(
    provider="groq",
    api_key=token
)

model = "openai/gpt-oss-120b"

topic = input("Enter a topic: ")

prompt = f"""
Create 5 study flashcards about {topic}.

For each flashcard:
1. Write a question.
2. Write a short answer.

Keep answers suitable for a second-year
computer science student.

Use this format:
Flashcard 1:
Question:
Answer:
"""

try:
    start_time = time.time()

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=700
    )

    end_time = time.time()
    latency = end_time - start_time

    print("\nGenerated Flashcards:\n")
    print(completion.choices[0].message.content)
    print(f"\nResponse time: {latency:.2f} seconds")

except Exception as e:
    print("\nSomething went wrong:")
    print(e)