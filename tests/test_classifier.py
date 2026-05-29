import sys
sys.path.insert(0, '.')
from app.retrieval.classifier import QueryClassifier

clf = QueryClassifier()

tests = [
    # (query, expected_top_collection)
    # Note: "997 rejection" is a troubleshooting signal — correct top.
    # edi_standards is returned as secondary — both will be searched.
    ("What does AK5*R mean in a 997 rejection?",              "troubleshooting"),
    ("What is the bullwhip effect in supply chain?",           "supply_chain_concepts"),
    ("How do I onboard a new AS2 trading partner?",            "integration_onboarding"),
    ("Why is my 856 ASN getting rejected?",                    "troubleshooting"),
    ("What is a SCAC code and who assigns it?",                "logistics_shipping"),
    ("What are Walmart's EDI compliance requirements?",        "compliance_regulations"),
    ("What does Cleo Integration Cloud support?",              "cleo_company"),
    # uploaded_documents gets priority bump when "this document" signal fires
    ("What does this document say about transaction sets?",    "uploaded_documents"),
    ("Tell me something about supply chain and EDI together",  None),  # multi-domain
]

print("=== Classifier routing test ===\n")
all_passed = True

for query, expected_top in tests:
    results = clf.classify(query)
    top_col, top_conf = results[0]
    collections_returned = [c for c, _ in results]

    if expected_top is None:
        # Multi-domain: just print, no assertion
        status = "INFO"
    elif top_col == expected_top:
        status = "PASS"
    else:
        status = "FAIL"
        all_passed = False

    print(f"[{status}] {query[:55]:<55}")
    print(f"       Top: {top_col} ({top_conf:.0%})")
    if len(results) > 1:
        others = ", ".join(f"{c}({s:.0%})" for c, s in results[1:])
        print(f"       Also: {others}")
    print()

print("classifier.py OK" if all_passed else "SOME TESTS FAILED — check routing")
