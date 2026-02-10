from app.core.reviewer import review_code

code = """
def add(a, b):
    return a + b
"""

result = review_code(code)
print(result)
