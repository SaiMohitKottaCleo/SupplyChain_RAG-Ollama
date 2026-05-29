"""
Synthetic corpus generator for Cleo Supply Chain Intelligence.

Uses the Claude API (claude-haiku-4-5 — fast + cheap) to generate knowledge
base articles for all 8 collections. Each article is a ~500-800 word factual
reference document — not Q&A, not bullet points, but prose that chunks well.

Usage:
    python scripts/generate_corpus.py
    python scripts/generate_corpus.py --collection edi_standards
    python scripts/generate_corpus.py --dry-run     (show topics, don't call API)

Requirements:
    - ANTHROPIC_API_KEY environment variable (or in .env)
    - pip install anthropic (already in requirements.txt)

Output: writes .txt files to data/raw/<collection>/ then ingests them.
"""
import os
import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import anthropic
from app.ingestion.pipeline import IngestionPipeline
from app.config import DATA_DIR
from loguru import logger


# ── Topic definitions ─────────────────────────────────────────────────────────
# Each entry: (collection_name, topic_title, writing_prompt_detail)
# We generate one article per topic. 10-15 topics per collection = solid coverage.

TOPICS = [

    # ── EDI Standards ─────────────────────────────────────────────────────────
    ("edi_standards", "X12 EDI 850 Purchase Order",
     "Explain the X12 850 Purchase Order transaction set: its purpose, structure, "
     "mandatory segments (BEG, DTM, PO1, CTT, SE), common loop structures, and how "
     "trading partners use it in procurement workflows."),

    ("edi_standards", "X12 EDI 856 Ship Notice",
     "Explain the X12 856 ASN (Advance Ship Notice): its purpose in supply chain "
     "visibility, the hierarchical HL loop structure (shipment/order/pack/item levels), "
     "key segments (BSN, TD1, TD5, REF, DTM), and why retailers mandate it."),

    ("edi_standards", "X12 EDI 810 Invoice",
     "Explain the X12 810 Invoice transaction set: structure, key segments (BIG, N1, "
     "IT1, TDS, SE), how it maps to a paper invoice, common validation rules, and "
     "the role of the 997 FA in confirming receipt."),

    ("edi_standards", "X12 EDI 997 Functional Acknowledgment",
     "Explain the 997 FA in depth: AK1 (functional group), AK2 (transaction set), "
     "AK3 (segment errors), AK4 (element errors), AK5 acceptance/rejection codes "
     "(A=accepted, E=accepted with errors, R=rejected, P=partial), AK9 summary. "
     "Include what AK5*R and AK5*E mean operationally."),

    ("edi_standards", "X12 EDI 940 Warehouse Shipping Order",
     "Explain the 940 Warehouse Shipping Order: when it is used (3PL/warehouse "
     "instructions), key segments, relationship to the 945 Warehouse Shipping Advice, "
     "and typical 3PL integration patterns."),

    ("edi_standards", "X12 EDI 945 Warehouse Shipping Advice",
     "Explain the 945 Warehouse Shipping Advice: how a 3PL confirms shipment back to "
     "a retailer or brand, key segments, and how it closes the loop with the 940."),

    ("edi_standards", "X12 EDI Envelope Structure: ISA, GS, ST, SE, GE, IEA",
     "Explain the X12 interchange envelope: ISA/IEA (interchange), GS/GE (functional "
     "group), ST/SE (transaction set). Include ISA segment element positions (ISA06 "
     "sender, ISA08 receiver, ISA13 control number), how control numbers work, and "
     "what happens when they mismatch."),

    ("edi_standards", "EDIFACT vs X12 EDI Standards",
     "Compare EDIFACT (UN/EDIFACT) and ANSI X12: origins, geographic dominance "
     "(EDIFACT in Europe/Asia, X12 in North America), segment syntax differences, "
     "message naming conventions, and how Cleo handles both."),

    ("edi_standards", "EDI Delimiters and Special Characters",
     "Explain EDI delimiters: ISA element separator (position 4), component separator "
     "(ISA16), segment terminator (end of ISA), how they are set per interchange, "
     "and common issues when data contains the delimiter character."),

    ("edi_standards", "X12 EDI 204 Motor Carrier Load Tender",
     "Explain the 204 Load Tender: purpose in freight procurement, key segments "
     "(B2, B2A, L11, AT5, LAD, S5, L3), stop-off loop structure, and integration "
     "with TMS systems."),

    # ── Supply Chain Concepts ─────────────────────────────────────────────────
    ("supply_chain_concepts", "Supply Chain Visibility and Real-Time Tracking",
     "Explain supply chain visibility: what it means, in-transit vs warehouse "
     "visibility, the role of EDI ASNs and IoT sensors, KPIs like OTIF (on-time "
     "in-full), and how visibility reduces safety stock requirements."),

    ("supply_chain_concepts", "Demand Forecasting Methods",
     "Explain demand forecasting in supply chains: statistical methods (moving "
     "average, exponential smoothing, ARIMA), machine learning approaches, the "
     "bullwhip effect and how forecasting accuracy reduces it, and common KPIs."),

    ("supply_chain_concepts", "Inventory Management: EOQ, Safety Stock, Reorder Points",
     "Explain inventory optimization: Economic Order Quantity formula, safety stock "
     "calculation (service level, demand variability, lead time), reorder point "
     "formula, ABC analysis, and how EDI enables vendor-managed inventory (VMI)."),

    ("supply_chain_concepts", "3PL, 4PL and Logistics Outsourcing",
     "Explain logistics outsourcing tiers: 1PL through 4PL, what a 3PL provides "
     "(warehousing, fulfillment, freight), what a 4PL adds (orchestration layer), "
     "typical EDI integrations required (940/945, 856, 210), and how to evaluate 3PLs."),

    ("supply_chain_concepts", "Purchase Order Lifecycle",
     "Walk through the full PO lifecycle: requisition → PO creation (EDI 850) → "
     "PO acknowledgment (855) → ASN (856) → receipt → invoice (810) → payment. "
     "Explain where EDI fits at each step and common failure points."),

    # ── Integration & Onboarding ──────────────────────────────────────────────
    ("integration_onboarding", "Trading Partner Onboarding Process",
     "Explain the end-to-end process of onboarding a new EDI trading partner: "
     "gathering ISA IDs and qualifiers, agreeing on transaction sets and versions, "
     "connectivity (AS2, SFTP, VAN), map development, testing (ISA test flag), "
     "and production cutover. Include common blockers and timelines."),

    ("integration_onboarding", "AS2 Protocol for EDI Transmission",
     "Explain AS2 (Applicability Statement 2): how it uses HTTPS + S/MIME for "
     "secure EDI transmission, MDN acknowledgments (synchronous vs async), "
     "certificate exchange, and how Cleo Clarify handles AS2 endpoints."),

    ("integration_onboarding", "SFTP vs AS2 vs VAN for EDI Connectivity",
     "Compare EDI connectivity options: SFTP (simple, polling-based), AS2 "
     "(real-time, MDN receipts, widely mandated by retailers), VAN (value-added "
     "network, older model with per-kilo billing). When to use each and migration paths."),

    ("integration_onboarding", "EDI Map Development and Testing",
     "Explain EDI mapping: translating between internal formats (ERP flat files, "
     "XML, JSON) and X12/EDIFACT. Cover segment loops, conditional elements, "
     "code list validation, the role of ISA test indicator (T vs P), and how "
     "to run end-to-end tests before going live."),

    ("integration_onboarding", "Cleo Clarify Integration Platform Overview",
     "Describe what Cleo Clarify does: B2B integration platform supporting EDI, "
     "API, and file-based integrations. Key capabilities: trading partner management, "
     "document transformation, visibility dashboard, exception management, "
     "and the difference between Cleo's cloud and on-premise deployments."),

    # ── Troubleshooting ───────────────────────────────────────────────────────
    ("troubleshooting", "Common EDI 997 Rejection Reasons and Fixes",
     "List and explain the most common reasons a 997 FA returns AK5*R: missing "
     "mandatory segments, invalid code values, segment count mismatch (AK2/SE), "
     "control number duplicates, wrong ISA version, and malformed delimiters. "
     "Include diagnostic steps and fixes for each."),

    ("troubleshooting", "EDI Transmission Failures: AS2 and SFTP Errors",
     "Explain common EDI transmission errors: AS2 MDN failures (certificate expired, "
     "wrong URL, firewall block), SFTP authentication failures (key rotation, "
     "IP whitelist), file naming convention mismatches, and file pickup timing issues. "
     "Include a diagnostic checklist."),

    ("troubleshooting", "EDI 850 Rejection by Trading Partner",
     "Explain why an 850 PO might be rejected by a retailer or trading partner: "
     "missing required qualifiers, invalid ship-to/bill-to identifiers, item not "
     "in catalog, date format errors, and duplicate PO numbers. Steps to diagnose "
     "and resubmit."),

    ("troubleshooting", "Duplicate Document and Control Number Issues",
     "Explain EDI control number management: ISA13 interchange control number, "
     "GS06 group control number, ST02 transaction control number. What happens "
     "when a duplicate is received, how trading partners handle them differently, "
     "and how to safely resubmit a rejected document with a new control number."),

    ("troubleshooting", "EDI Character Encoding and Special Character Issues",
     "Explain encoding problems in EDI: ASCII vs UTF-8 vs EBCDIC, what happens "
     "when non-ASCII characters appear in EDI data (accented names, foreign "
     "addresses), how to detect and strip invalid characters, and best practices "
     "for encoding at integration boundaries."),

    # ── Logistics & Shipping ──────────────────────────────────────────────────
    ("logistics_shipping", "Incoterms 2020: Complete Reference",
     "Explain all 11 Incoterms 2020 rules: EXW, FCA, CPT, CIP, DAP, DPU, DDP "
     "(any mode), and FAS, FOB, CFR, CIF (sea only). For each: who bears risk, "
     "who pays freight, where risk transfers. Common mistakes and how Incoterms "
     "appear in EDI 850/856 documents."),

    ("logistics_shipping", "Freight Modes: FTL, LTL, Parcel, Intermodal",
     "Compare freight modes: Full Truckload (FTL), Less-than-Truckload (LTL), "
     "parcel carriers (UPS/FedEx/USPS), and intermodal (rail+truck). Cost drivers, "
     "transit times, when to use each, and EDI transaction sets for each mode "
     "(204, 210, 214, 856)."),

    ("logistics_shipping", "Carrier Tracking and EDI 214 Shipment Status",
     "Explain the EDI 214 Transportation Carrier Shipment Status Message: "
     "when carriers send it, key status codes (AT7 shipment status), how shippers "
     "use it for in-transit visibility, and how it integrates with TMS systems "
     "and customer-facing tracking portals."),

    ("logistics_shipping", "Customs and Import/Export Documentation",
     "Explain key customs documents: Commercial Invoice, Packing List, Bill of "
     "Lading, Certificate of Origin, HS codes, and how EDI supports cross-border "
     "trade (EDIFACT CUSCAR, X12 309). Common clearance delays and how accurate "
     "EDI data prevents them."),

    # ── Compliance & Regulations ──────────────────────────────────────────────
    ("compliance_regulations", "GDPR and Data Privacy in Supply Chain EDI",
     "Explain GDPR implications for supply chain EDI: what personal data appears "
     "in EDI documents (names, addresses, contact info), data retention obligations, "
     "the right to erasure challenge in immutable EDI archives, and best practices "
     "for privacy-compliant EDI infrastructure."),

    ("compliance_regulations", "FDA DSCSA Drug Supply Chain Security Act",
     "Explain DSCSA requirements for pharmaceutical supply chains: serialization "
     "at unit level (SNDC), EPCIS for track-and-trace, the 2023 interoperability "
     "deadline, and how EDI 856 ASNs carry serial number information for compliance."),

    ("compliance_regulations", "GS1 Standards: GTIN, GLN, SSCC Barcodes",
     "Explain GS1 standards used in supply chain: GTIN (product identification), "
     "GLN (location identification), SSCC (pallet/case serial numbers), GS1-128 "
     "barcodes, and how they map to EDI fields in 850, 856, and 945 transactions."),

    ("compliance_regulations", "Retail Compliance: Chargeback Prevention",
     "Explain retailer compliance programs: why retailers issue chargebacks "
     "(labeling errors, ASN timing, carton count mismatches, ticket violations), "
     "common chargeback categories (routing guide violations, EDI errors, packaging), "
     "and how EDI accuracy directly reduces chargeback exposure."),

    # ── Cleo Company ─────────────────────────────────────────────────────────
    ("cleo_company", "Cleo Clarify: Platform Architecture and Capabilities",
     "Describe Cleo Clarify's technical architecture: multi-protocol support "
     "(AS2, SFTP, HTTPS, FTP), EDI translation engine, API integration layer, "
     "real-time visibility dashboard, exception management workflow, and the "
     "Cleo Integration Cloud ecosystem."),

    ("cleo_company", "Cleo EDI Network and Trading Partner Community",
     "Explain Cleo's pre-built trading partner network: the number of pre-configured "
     "connections, major retail and 3PL partners already in the network, what "
     "'pre-built' means (existing maps, tested connectivity), and how this reduces "
     "onboarding time vs. building connections from scratch."),

    ("cleo_company", "Cleo vs. Traditional VAN EDI Providers",
     "Compare Cleo's approach to legacy VAN providers (Sterling Commerce/IBM, "
     "Inovis/OpenText, SPS Commerce): pricing model differences (per-transaction "
     "vs. subscription), modern API capabilities VANs lack, visibility features, "
     "and migration considerations for companies switching from VAN to Cleo."),
]


def generate_article(client: anthropic.Anthropic, topic: str, detail: str) -> str:
    """Call Claude to generate one knowledge base article."""
    prompt = f"""Write a detailed, factual reference article about: {topic}

Topic guidance: {detail}

Requirements:
- 500-800 words of dense, accurate technical content
- Written as a reference document, not Q&A or bullet points
- Include specific details: segment names, code values, field names, formulas where relevant
- Assume the reader is a supply chain or EDI professional
- No introductory fluff like "In today's supply chain..." — start directly with substance
- Do not use headers or markdown formatting — write flowing paragraphs

Write the article now:"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def run(dry_run: bool = False, target_collection: str = None):
    topics = TOPICS
    if target_collection:
        topics = [(c, t, d) for c, t, d in TOPICS if c == target_collection]
        if not topics:
            print(f"No topics for collection '{target_collection}'")
            sys.exit(1)

    print(f"\n{'DRY RUN — ' if dry_run else ''}Generating {len(topics)} article(s):\n")
    for col, title, _ in topics:
        print(f"  [{col:30s}]  {title}")

    if dry_run:
        print("\n(Dry run — nothing generated. Remove --dry-run to generate.)")
        return

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("\nERROR: ANTHROPIC_API_KEY not set.")
        print("Add it to your .env file: ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    client   = anthropic.Anthropic(api_key=api_key)
    pipeline = IngestionPipeline()
    raw_dir  = DATA_DIR / "raw"

    ok = 0
    errors = 0
    for i, (collection, title, detail) in enumerate(topics, 1):
        print(f"\n[{i:2d}/{len(topics)}] Generating: {title[:60]}...")
        try:
            text = generate_article(client, title, detail)

            # Save to disk (for reference / re-ingestion)
            out_dir = raw_dir / collection
            out_dir.mkdir(parents=True, exist_ok=True)
            safe_name = title.lower().replace(" ", "_")[:60].replace("/", "-")
            out_path  = out_dir / f"{safe_name}.txt"
            out_path.write_text(f"{title}\n\n{text}", encoding="utf-8")

            # Ingest immediately
            result = pipeline.ingest(
                source=str(out_path),
                collection_name=collection,
                extra_metadata={"source": title, "source_type": "synthetic"},
            )

            if result["status"] == "ok":
                ok += 1
                print(f"         ✓  {result['chunks_stored']} chunk(s) stored")
            else:
                errors += 1
                print(f"         ✗  Ingest error: {result.get('error')}")

            # Rate limit: ~1 req/sec to stay well within Haiku limits
            time.sleep(1)

        except Exception as e:
            errors += 1
            logger.error(f"Failed: {title}: {e}")
            print(f"         ✗  {e}")

    print(f"\n{'─'*60}")
    print(f"  Done: {ok} articles generated, {errors} failed")
    print(f"  Data saved to: {raw_dir}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic corpus for CSCI RAG")
    parser.add_argument("--dry-run", action="store_true",
                        help="List topics without calling the API")
    parser.add_argument("--collection", type=str, default=None,
                        help="Only generate for this collection")
    args = parser.parse_args()
    run(dry_run=args.dry_run, target_collection=args.collection)
