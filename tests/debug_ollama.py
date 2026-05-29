import sys
sys.path.insert(0, '.')
import ollama

print("=== ollama.list() debug ===")
try:
    r = ollama.list()
    print(f"Response type : {type(r)}")
    print(f"Raw response  : {r}")
    print()

    # Try both APIs — object style and dict style
    if hasattr(r, 'models'):
        print(f"models (attr) : {r.models}")
        for m in r.models:
            print(f"  model object: {m}")
            print(f"  type        : {type(m)}")
            if hasattr(m, 'model'):
                print(f"  .model attr : {m.model}")
            if isinstance(m, dict):
                print(f"  dict keys   : {m.keys()}")
    elif isinstance(r, dict):
        print(f"dict keys     : {r.keys()}")
        print(f"models key    : {r.get('models', [])}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("=== Quick chat test ===")
try:
    resp = ollama.chat(
        model="phi3:mini",
        messages=[{"role": "user", "content": "Reply with just: OK"}],
        options={"num_predict": 5}
    )
    print(f"Response type : {type(resp)}")
    print(f"Has .message  : {hasattr(resp, 'message')}")
    if hasattr(resp, 'message'):
        print(f"Content       : {resp.message.content}")
    else:
        print(f"Raw           : {resp}")
except Exception as e:
    print(f"Chat error    : {e}")
