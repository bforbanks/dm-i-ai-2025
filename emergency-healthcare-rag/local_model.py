import ollama

model = 'gemma3:12b'
system_prompt = "You are a caveman. Keep your response somewhat short."

print()

messages = [
    {
        'role': 'system',
        'content': system_prompt
    },
]

while True:
    print()
    content = input("User: ")
    messages.append(
        {
            'role': 'user',
            'content': content
        }
    )

    response = ollama.chat(model=model, messages=messages)

    messages.append(
        {
            'role': 'assistant',
            'content': response.message.content
        }
    )

    print("AI: " + response.message.content)