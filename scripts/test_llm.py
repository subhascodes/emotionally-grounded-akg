from scripts.llm_backend import generate_text

print("Testing LLM...")
out = generate_text("Say hello in one sentence.")
print("Output:", out)