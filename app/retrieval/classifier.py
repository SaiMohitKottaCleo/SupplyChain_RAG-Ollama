from typing import List, Tuple

# ── Signal dictionaries ────────────────────────────────────────────────────
# Each key = collection name. Values = keywords/phrases that strongly
# indicate a query belongs to that collection.
#
# Scoring: each match adds len(signal.split()) to the score.
# Rationale: "997 functional acknowledgment" (3 words) is stronger evidence
# than "edi" (1 word). Longer matches = higher specificity.

COLLECTION_SIGNALS = {
    "edi_standards": [
        "x12", "edi", "transaction set", "segment", "loop", "envelope",
        "isa", "gs segment", "st segment", "se segment", "ge segment",
        "850", "856", "810", "997", "999", "820", "832", "846",
        "940", "945", "214", "753", "754",
        "functional acknowledgment", "interchange", "delimiter",
        "ak1", "ak2", "ak5", "ak9", "ik3", "ik4",
        "element", "composite element", "data element",
        "qualifier", "control number", "functional group",
        "transaction set identifier", "implementation acknowledgment",
    ],
    "supply_chain_concepts": [
        "supply chain", "inventory", "demand", "forecast", "warehouse",
        "fulfillment", "bullwhip", "bullwhip effect",
        "vendor managed", "vmi", "just in time", "jit",
        "safety stock", "lead time", "purchase order", "procurement",
        "sourcing", "distribution", "replenishment", "sku",
        "stockout", "backorder", "reorder point", "economic order quantity",
        "eoq", "abc analysis", "cycle counting", "3pl", "4pl",
        "cross dock", "cross-docking", "last mile",
    ],
    "integration_onboarding": [
        "onboard", "onboarding", "trading partner", "setup", "configure",
        "connection", "mapping", "map", "go live", "certification",
        "partner agreement", "isa qualifier", "as2", "as4",
        "sftp", "ftp", "ftps", "van", "value added network",
        "mailbox", "communication protocol", "test transaction",
        "pilot testing", "production cutover",
    ],
    "troubleshooting": [
        "error", "reject", "rejection", "fail", "failure", "issue",
        "problem", "debug", "fix", "wrong", "invalid", "missing",
        "duplicate", "why", "not working", "broken", "cannot",
        "997 rejection", "999 rejection", "ak5*r", "ak5*e",
        "mismatch", "terminator", "control number error",
        "segment not found", "mandatory element", "out of sequence",
    ],
    "logistics_shipping": [
        "shipping", "freight", "carrier", "bill of lading", "bol",
        "scac", "scac code", "ltl", "ftl", "drayage", "dwell",
        "detention", "gs1", "gs1-128", "ucc128", "ucc-128",
        "sscc", "pallet", "label", "tracking", "delivery", "transit",
        "dock appointment", "freight class", "nmfc", "density",
    ],
    "compliance_regulations": [
        "compliance", "regulation", "fsma", "dscsa", "hipaa", "fda",
        "walmart", "target", "amazon", "costco", "kroger",
        "requirement", "mandate", "retail compliance",
        "routing guide", "vendor guide", "chargeback",
        "food safety", "drug supply chain", "serialization",
        "traceability", "lot tracking",
    ],
    "cleo_company": [
        "cleo", "cleo integration", "cic", "cleo harmony",
        "cleo integration cloud", "does cleo", "can cleo",
        "cleo support", "cleo allow", "cleo product", "cleo feature",
        "what does cleo", "cleo platform", "cleo cloud",
        "cleo accelerate", "cleo network",
    ],
    "uploaded_documents": [
        "this document", "the document", "uploaded document",
        "attached file", "this file", "the file",
        "this spec", "this specification", "trading partner spec",
        "this guide", "this pdf",
    ],
}


class QueryClassifier:
    """
    Routes incoming queries to the most relevant ChromaDB collection(s).

    Algorithm:
      1. Lowercase the query
      2. For each collection, sum keyword match weights
         (weight = number of words in the matched signal phrase)
      3. Sort collections by score descending
      4. If top score is 0 → query is ambiguous → return all collections
      5. Normalize scores to [0,1] relative to top score
      6. Return collections scoring ≥ 30% of top (multi-collection for
         queries that span domains)
      7. Always include 'uploaded_documents' if it scored > 0

    Returns: List of (collection_name, confidence) tuples, sorted by confidence.
    """

    def classify(self, query: str) -> List[Tuple[str, float]]:
        query_lower = query.lower()
        scores: dict = {}

        for collection, signals in COLLECTION_SIGNALS.items():
            score = 0
            for signal in signals:
                if signal in query_lower:
                    # Multi-word signals score proportionally higher
                    score += len(signal.split())
            scores[collection] = score

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Ambiguous query (no keyword matched) → search everything
        if ranked[0][1] == 0:
            return [(col, 0.1) for col, _ in ranked]

        # Normalize relative to top scorer
        max_score = ranked[0][1]
        normalized = [(col, round(score / max_score, 3)) for col, score in ranked]

        # Keep collections scoring ≥ 30% of top
        relevant = [(col, conf) for col, conf in normalized if conf >= 0.3]

        # uploaded_documents gets a priority bump when it matched:
        # "this document / the document" is an unambiguous signal the user
        # uploaded something and wants to query it — always surface it first.
        ud_score = scores.get("uploaded_documents", 0)
        if ud_score > 0:
            # Remove from current position (wherever it landed)
            relevant = [(c, s) for c, s in relevant if c != "uploaded_documents"]
            # Prepend it with its normalized confidence
            ud_conf = round(ud_score / max_score, 3)
            relevant.insert(0, ("uploaded_documents", ud_conf))

        return relevant
