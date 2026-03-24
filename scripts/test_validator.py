from akg.transition_validator import validate_sequence, get_allowed_next

tests = [
    ["distress", "anger", "gratitude"],   # likely partially valid
    ["joy", "distress"],                  # should be invalid
    ["fear", "hope", "joy"],              # should be valid
    ["anger", "joy"],                     # should be invalid
    ["surprise", "joy"],                  # valid
    ["unknown", "joy"],                   # invalid
    ["hope"],                             # single element
]

for seq in tests:
    print(f"\nSequence: {seq}")
    result = validate_sequence(seq)
    print(result)