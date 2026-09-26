import html
import os
import re
import time
import xml.etree.ElementTree as ET
from functools import lru_cache
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor
from app.services.colbert_reranker import rerank_candidates

import requests



MEDLINEPLUS_URL = "https://wsearch.nlm.nih.gov/ws/query"
DAILYMED_SPLS_URL = (
    "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
)
DAILYMED_SPL_XML = (
    "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml"
)
OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
RXNORM_RXCUI_URL = "https://rxnav.nlm.nih.gov/REST/rxcui.json"

PUBMED_SEARCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
)
PUBMED_FETCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
)


def clean_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



def extract_labeled_field(
    query: str,
    label: str,
) -> Optional[str]:
    """
    Extract fields such as Subject, Drug, and Question
    from either multiline or flattened queries.

    Examples supported:

    Subject: Noonan syndrome
    Question: ...

    Subject: Noonan syndrome Question: ...
    """

    pattern = (
        rf"(?is)(?:^|\s)"
        rf"{re.escape(label)}\s*:\s*"
        rf"(.*?)"
        rf"(?=\s+(?:Subject|Drug|Question)\s*:|$)"
    )

    match = re.search(pattern, query)

    if not match:
        return None

    value = clean_text(match.group(1))

    return value or None

def extract_ndc(query: str) -> Optional[str]:
    match = re.search(
        r"\b\d{4,5}-\d{3,4}-\d{1,2}\b",
        query,
    )
    return match.group(0) if match else None


def extract_drug_hint(query: str) -> Optional[str]:
    # 1. Explicit MedicationQA Drug field.
    drug_field = extract_labeled_field(
        query,
        "Drug",
    )

    if drug_field:
        if not drug_field.lower().startswith("n/a"):
            return drug_field

    # 2. Drug name following an NDC.
    ndc_match = re.search(
        r"(?i)NDC\s*#?\s*:?\s*"
        r"\d{4,5}-\d{3,4}-\d{1,2}"
        r"[\s,;:-]+"
        r"([A-Za-z][A-Za-z0-9\-]*)",
        query,
    )

    if ndc_match:
        return clean_text(
            ndc_match.group(1)
        )

    # 3. LiveQA subject may identify the drug.
    subject = extract_labeled_field(
        query,
        "Subject",
    )

    if subject:
        subject_lower = subject.lower()

        if re.search(
            r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml)\b",
            subject_lower,
        ):
            candidate = re.sub(
                r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml)\b.*$",
                "",
                subject,
                flags=re.IGNORECASE,
            ).strip()

            if candidate:
                return candidate

    # 4. CamelCase product names.
    question = (
        extract_labeled_field(
            query,
            "Question",
        )
        or query
    )

    camel = re.findall(
        r"\b[A-Z][a-z]+[A-Z][A-Za-z0-9]*\b",
        question,
    )

    if camel:
        return camel[0]

    return None


@lru_cache(maxsize=2048)
def _is_rxnorm_drug(term: str) -> bool:
    """Confirm a possible drug name using the official RxNorm API."""
    try:
        response = requests.get(
            RXNORM_RXCUI_URL,
            params={
                "name": term,
                "search": 2,
                "allsrc": 0,
            },
            timeout=8,
        )
        response.raise_for_status()

        ids = (
            response.json()
            .get("idGroup", {})
            .get("rxnormId", [])
        )

        return bool(ids)

    except Exception:
        return False


def extract_drug_hints(
    query: str,
    max_drugs: int = 3,
) -> list[str]:
    """
    Extract and verify drug names from ordinary free-text queries.

    Existing labeled fields, NDC patterns, and product-name rules
    remain supported through extract_drug_hint().
    """
    hints: list[str] = []
    candidates: list[str] = []

    legacy_hint = extract_drug_hint(query)

    if legacy_hint:
        hints.append(legacy_hint)

    question = (
        extract_labeled_field(query, "Question")
        or query
    )

    cue_patterns = [
        (
            r"(?i)\b(?:take|takes|taking|took|"
            r"start|starts|started|starting|"
            r"stop|stops|stopped|stopping|"
            r"use|uses|using|used|"
            r"prescribe|prescribed|"
            r"switch|switched|switching)\s+"
            r"([A-Za-z][A-Za-z-]{3,})"
        ),
        (
            r"(?i)\b([A-Za-z][A-Za-z-]{3,})\s+"
            r"\d+(?:\.\d+)?\s*"
            r"(?:mg|mcg|g|ml)\b"
        ),
    ]

    for pattern in cue_patterns:
        candidates.extend(
            re.findall(pattern, question)
        )

    # Also scan a limited number of individual words.
    # This handles phrases such as "started taking methadone"
    # when the cue pattern first captures a non-drug word.
    candidates.extend(
        re.findall(
            r"\b[A-Za-z][A-Za-z-]{4,}\b",
            question.lower(),
        )[:12]
    )

    ignored = {
        "about",
        "after",
        "allergic",
        "because",
        "before",
        "could",
        "doctor",
        "having",
        "information",
        "medicine",
        "medication",
        "please",
        "should",
        "started",
        "taking",
        "there",
        "these",
        "those",
        "using",
        "would",
    }

    seen = {
        clean_text(item).lower()
        for item in hints
    }

    for candidate in candidates:
        candidate = clean_text(candidate).lower()

        if (
            not candidate
            or candidate in ignored
            or candidate in seen
        ):
            continue

        if _is_rxnorm_drug(candidate):
            hints.append(candidate)
            seen.add(candidate)

        if len(hints) >= max_drugs:
            break

    return hints


def _doc(
    content: str,
    source: str,
    url: str,
) -> dict[str, Any]:
    return {
        "content": clean_text(content),
        "category": "trusted_web_medical",
        "document_source": source,
        "document_url": url,
        "relevance_score": 0,
        "evidence_origin": "trusted_web",
    }


def search_medlineplus(
    query: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    # Subject is useful for LiveQA queries.
    subject = extract_labeled_field(
        query,
        "Subject",
    )

    search_term = (
        subject
        if subject
        else clean_text(query)
    )

    try:
        response = requests.get(
            MEDLINEPLUS_URL,
            params={
                "db": "healthTopics",
                "term": search_term,
                "rettype": "brief",
                "retmax": limit,
                "tool": "medical_rag_thesis",
            },
            timeout=20,
        )
        response.raise_for_status()

        root = ET.fromstring(response.text)
        docs = []

        for document in root.findall(".//document"):
            url = document.attrib.get("url", "")
            values = {}

            for node in document.findall("content"):
                name = node.attrib.get("name", "")
                values[name] = "".join(
                    node.itertext()
                ).strip()

            content = " ".join(
                x
                for x in [
                    values.get("title", ""),
                    values.get("FullSummary", ""),
                    values.get("snippet", ""),
                ]
                if x
            )

            if content:
                docs.append(
                    _doc(
                        content=content,
                        source="MedlinePlus",
                        url=url,
                    )
                )

        return docs[:limit]

    except Exception:
        return []



def extract_dosage_form(query: str) -> Optional[str]:
    text = query.lower()

    forms = [
        ("nasal spray", "nasal spray"),
        ("spray", "spray"),
        ("tablets", "tablet"),
        ("tablet", "tablet"),
        ("capsules", "capsule"),
        ("capsule", "capsule"),
        ("injection", "injection"),
        ("injectable", "injection"),
        ("ointment", "ointment"),
        ("cream", "cream"),
        ("inhaler", "inhaler"),
        ("patch", "patch"),
        ("solution", "solution"),
        ("suspension", "suspension"),
        ("syrup", "syrup"),
    ]

    for phrase, normalized in forms:
        if phrase in text:
            return normalized

    return None


def matches_dosage_form(
    title: str,
    content: str,
    requested_form: Optional[str],
) -> bool:
    if not requested_form:
        return True

    title_lower = clean_text(title).lower()
    content_lower = clean_text(content).lower()

    # Prefer the DailyMed product title because the full
    # label may mention other dosage forms in neighboring text.
    target = title_lower or content_lower

    if requested_form == "tablet":
        if "spray" in target:
            return False

        if "orally disintegrating" in target:
            return False

        return "tablet" in target

    if requested_form == "capsule":
        return (
            "capsule" in target
            and "tablet" not in target
            and "spray" not in target
        )

    if requested_form == "nasal spray":
        return "nasal spray" in target

    if requested_form == "spray":
        return "spray" in target

    return requested_form in target


def search_dailymed(
    query: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    ndc = extract_ndc(query)
    drug = extract_drug_hint(query)
    dosage_form = extract_dosage_form(query)

    searches = []

    if ndc:
        searches.append(
            ("exact_ndc", {"ndc": ndc})
        )

    if drug:
        searches.append(
            (
                "drug_name",
                {
                    "drug_name": drug,
                    "name_type": "both",
                },
            )
        )

    for match_method, search_params in searches:
        try:
            response = requests.get(
                DAILYMED_SPLS_URL,
                params={
                    **search_params,
                    "pagesize": limit,
                },
                timeout=20,
            )
            response.raise_for_status()

            items = response.json().get("data", [])

            if not items:
                continue

            docs = []

            for item in items:
                setid = item.get("setid")
                title = clean_text(item.get("title"))

                if not setid:
                    continue

                xml_url = DAILYMED_SPL_XML.format(
                    setid=setid
                )

                xml_response = requests.get(
                    xml_url,
                    timeout=20,
                )
                xml_response.raise_for_status()

                root = ET.fromstring(
                    xml_response.content
                )

                content = clean_text(
                    " ".join(root.itertext())
                )[:18000]

                if not matches_dosage_form(
                    title=title,
                    content=content,
                    requested_form=dosage_form,
                ):
                    continue

                if not content:
                    continue

                doc = _doc(
                    content=content,
                    source="DailyMed",
                    url=xml_url,
                )

                doc["product_title"] = title

                if ndc and match_method == "drug_name":
                    doc["match_method"] = "drug_name_fallback"
                    doc["exact_product_confirmed"] = False
                else:
                    doc["match_method"] = match_method
                    doc["exact_product_confirmed"] = (
                        match_method == "exact_ndc"
                    )

                doc["requested_ndc"] = ndc or ""
                doc["requested_drug"] = drug or ""
                doc["requested_dosage_form"] = (
                    dosage_form or ""
                )

                docs.append(doc)

            if docs:
                drug_key = (
                    clean_text(drug).lower()
                    if drug
                    else ""
                )

                # Prefer a single-ingredient label when the
                # query itself does not request a combination.
                if drug_key and " and " not in drug_key:
                    single_ingredient_docs = []

                    for doc in docs:
                        title = clean_text(
                            doc.get(
                                "product_title",
                                "",
                            )
                        ).lower()

                        product_name = title.split(
                            "[",
                            1,
                        )[0].strip()

                        if (
                            drug_key in product_name
                            and " and " not in product_name
                        ):
                            single_ingredient_docs.append(
                                doc
                            )

                    if single_ingredient_docs:
                        docs = single_ingredient_docs

                def dailymed_rank(
                    doc: dict[str, Any],
                ) -> int:
                    title = clean_text(
                        doc.get(
                            "product_title",
                            "",
                        )
                    ).lower()

                    product_name = title.split(
                        "[",
                        1,
                    )[0].strip()

                    score = 0

                    if (
                        drug_key
                        and product_name.startswith(
                            drug_key
                        )
                    ):
                        score += 10

                    if "tablet" in product_name:
                        score += 3
                    elif "capsule" in product_name:
                        score += 2
                    elif "solution" in product_name:
                        score += 1
                    elif "powder" in product_name:
                        score -= 1

                    return score

                docs.sort(
                    key=dailymed_rank,
                    reverse=True,
                )

                return docs

        except Exception:
            continue

    return []


def search_openfda(
    query: str,
    limit: int = 2,
) -> list[dict[str, Any]]:
    ndc = extract_ndc(query)
    drug = extract_drug_hint(query)

    searches = []

    if ndc:
        searches.append(
            f'openfda.package_ndc:"{ndc}"'
        )

    if drug:
        safe_drug = drug.replace('"', "")
        searches.extend(
            [
                f'openfda.generic_name:"{safe_drug}"',
                f'openfda.brand_name:"{safe_drug}"',
            ]
        )

    if not searches:
        return []

    fields = [
        "description",
        "active_ingredient",
        "inactive_ingredient",
        "indications_and_usage",
        "dosage_and_administration",
        "dosage_forms_and_strengths",
        "drug_interactions",
        "contraindications",
        "warnings",
        "adverse_reactions",
        "how_supplied",
        "clinical_pharmacology",
        "spl_product_data_elements",
    ]

    for search in searches:
        try:
            response = requests.get(
                OPENFDA_LABEL_URL,
                params={
                    "search": search,
                    "limit": limit,
                },
                timeout=20,
            )

            if response.status_code != 200:
                continue

            results = response.json().get(
                "results",
                [],
            )

            docs = []

            for result in results:
                parts = []

                for field in fields:
                    value = result.get(field)

                    if isinstance(value, list):
                        parts.extend(
                            clean_text(x)
                            for x in value
                            if clean_text(x)
                        )
                    elif value:
                        parts.append(
                            clean_text(value)
                        )

                content = " ".join(parts)[:18000]

                if not content:
                    continue

                set_id = result.get("set_id", "")

                url = (
                    "https://open.fda.gov/apis/drug/label/"
                )

                if set_id:
                    url += f"#set-id-{set_id}"

                docs.append(
                    _doc(
                        content=content,
                        source="openFDA Drug Label",
                        url=url,
                    )
                )

            if docs:
                return docs

        except Exception:
            continue

    return []



def is_reference_question(query: str) -> bool:
    text = query.lower()

    indicators = (
        "reference",
        "references",
        "article",
        "articles",
        "study",
        "studies",
        "research",
        "paper",
        "papers",
        "literature",
        "publication",
        "publications",
    )

    return any(
        indicator in text
        for indicator in indicators
    )



PUBMED_QUERY_STOPWORDS = {
    "what", "which", "who", "when", "where", "why", "how",
    "to", "the", "a", "an", "is", "are", "was", "were",
    "be", "been", "being",
    "do", "does", "did", "done",
    "can", "could", "should", "would", "may", "might",
    "of", "for", "with", "about", "in", "on", "at",
    "by", "from", "as", "and", "or",
    "patient", "patients",
    "diagnose", "diagnosis",
    "symptom", "symptoms",
    "percentage", "percent",
    "used", "use", "using",
    "classify", "classified", "classification",
    "having",
    "represented", "represents",
    "types", "type",
    "therapy", "therapies",
    "treat", "treated", "treatment",
    "research", "clinical", "trial", "trials",
}


def build_pubmed_entity_query(query: str) -> str:
    question = extract_labeled_field(
        query,
        "Question",
    )

    text = clean_text(
        question or query
    )

    # Remove short parenthetical acronyms, while preserving
    # the full medical entity appearing outside the parentheses.
    text = re.sub(
        r"\(\s*[A-Z0-9/-]{2,12}\s*\)",
        " ",
        text,
    )

    tokens = re.findall(
        r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*",
        text,
    )

    kept = [
        token
        for token in tokens
        if token.lower()
        not in PUBMED_QUERY_STOPWORDS
    ]

    return " ".join(kept).strip()


def build_pubmed_query_variants(
    query: str,
) -> list[str]:
    variants = []

    original = build_pubmed_query(query)

    if original:
        variants.append(original)

    entity_query = build_pubmed_entity_query(
        query
    )

    if (
        entity_query
        and entity_query not in variants
    ):
        variants.append(entity_query)

    tokens = entity_query.split()

    # Relax only when the entity-focused query is long.
    # Try progressively shorter contiguous windows while
    # preserving local medical phrase structure.
    for size in [7, 6, 5, 4]:
        if len(tokens) <= size:
            continue

        for start in range(
            0,
            len(tokens) - size + 1,
        ):
            candidate = " ".join(
                tokens[start:start + size]
            )

            if candidate not in variants:
                variants.append(candidate)

    return variants


def build_pubmed_query(query: str) -> str:
    question = extract_labeled_field(
        query,
        "Question",
    )

    search_text = clean_text(
        question or query
    )

    patterns = [
        r"(?i)^what are the references (?:with|for|about)\s+",
        r"(?i)^what references (?:are there )?(?:with|for|about)\s+",
        r"(?i)^find references (?:with|for|about)\s+",
        r"(?i)^find articles (?:with|for|about)\s+",
        r"(?i)^find studies (?:with|for|about)\s+",
    ]

    for pattern in patterns:
        search_text = re.sub(
            pattern,
            "",
            search_text,
        ).strip()

    subject = extract_labeled_field(
        query,
        "Subject",
    )

    if (
        subject
        and subject.lower()
        not in search_text.lower()
    ):
        search_text = (
            f"{subject} AND {search_text}"
        )

    return search_text


def search_pubmed(
    query: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    original_term = build_pubmed_query(query)
    entity_term = build_pubmed_entity_query(query)

    strict_terms = []

    for term in [original_term, entity_term]:
        if term and term not in strict_terms:
            strict_terms.append(term)

    all_terms = build_pubmed_query_variants(query)

    relaxed_terms = [
        term
        for term in all_terms
        if term not in strict_terms
    ]

    if not strict_terms and not relaxed_terms:
        return []

    try:
        ids = []
        successful_search_term = None

        def _search_ids(
            search_term: str,
        ) -> list[str]:
            search_response = None
            last_search_error = None

            for attempt in range(3):
                try:
                    search_response = requests.get(
                        PUBMED_SEARCH_URL,
                        params={
                            "db": "pubmed",
                            "term": search_term,
                            "retmode": "json",
                            "retmax": limit,
                            "tool": "medical_rag_thesis",
                        },
                        headers={
                            "User-Agent":
                                "medical-rag-thesis/1.0"
                        },
                        timeout=20,
                    )

                    search_response.raise_for_status()
                    break

                except Exception as exc:
                    last_search_error = exc

                    if attempt < 2:
                        time.sleep(
                            1.0 * (attempt + 1)
                        )

            if search_response is None:
                print(
                    "PUBMED SEARCH ERROR:",
                    repr(last_search_error),
                )
                return []

            return (
                search_response.json()
                .get("esearchresult", {})
                .get("idlist", [])
            )

        # First preserve the original behavior for the
        # complete query and the entity-focused query.
        for term_index, search_term in enumerate(
            strict_terms
        ):
            term_ids = _search_ids(search_term)

            if term_ids:
                ids = term_ids[:limit]
                successful_search_term = search_term

                if term_index > 0:
                    print(
                        "PUBMED QUERY RELAXED:",
                        repr(strict_terms[0]),
                        "->",
                        repr(search_term),
                    )

                break

            if term_index < len(strict_terms) - 1:
                time.sleep(0.4)

        # Only if both strict queries fail, evaluate all
        # controlled relaxed windows instead of accepting
        # the first successful window.
        if not ids and relaxed_terms:
            relaxed_results = []

            for term_index, search_term in enumerate(
                relaxed_terms
            ):
                term_ids = _search_ids(search_term)

                if term_ids:
                    relaxed_results.append(
                        (search_term, term_ids)
                    )

                if term_index < len(relaxed_terms) - 1:
                    time.sleep(0.4)

            if relaxed_results:
                selected_ids = []
                seen_ids = set()

                max_rank = max(
                    len(term_ids)
                    for _, term_ids
                    in relaxed_results
                )

                # Round-robin selection prevents an early,
                # broad query window from occupying the
                # complete PubMed candidate set.
                for rank_index in range(max_rank):
                    for _, term_ids in relaxed_results:
                        if rank_index >= len(term_ids):
                            continue

                        pmid = term_ids[rank_index]

                        if pmid in seen_ids:
                            continue

                        seen_ids.add(pmid)
                        selected_ids.append(pmid)

                        if len(selected_ids) >= limit:
                            break

                    if len(selected_ids) >= limit:
                        break

                ids = selected_ids

                print(
                    "PUBMED QUERY RELAXED:",
                    repr(
                        strict_terms[0]
                        if strict_terms
                        else query
                    ),
                    "->",
                    f"{len(relaxed_results)} balanced variants",
                )

        if not ids:
            return []

        # Keep request rate conservative.
        time.sleep(0.4)

        fetch_response = None
        last_fetch_error = None

        for attempt in range(3):
            try:
                fetch_response = requests.get(
                    PUBMED_FETCH_URL,
                    params={
                        "db": "pubmed",
                        "id": ",".join(ids),
                        "retmode": "xml",
                        "tool": "medical_rag_thesis",
                    },
                    headers={
                        "User-Agent": "medical-rag-thesis/1.0"
                    },
                    timeout=20,
                )
                fetch_response.raise_for_status()
                break

            except Exception as exc:
                last_fetch_error = exc

                if attempt < 2:
                    time.sleep(1.0 * (attempt + 1))

        if fetch_response is None:
            print(
                "PUBMED FETCH ERROR:",
                repr(last_fetch_error),
            )
            return []

        root = ET.fromstring(
            fetch_response.content
        )

        docs = []

        for article in root.findall(
            ".//PubmedArticle"
        ):
            pmid = clean_text(
                article.findtext(
                    ".//PMID",
                    default="",
                )
            )

            title_node = article.find(
                ".//ArticleTitle"
            )

            title = (
                clean_text(
                    "".join(
                        title_node.itertext()
                    )
                )
                if title_node is not None
                else ""
            )

            abstract_parts = []

            for node in article.findall(
                ".//Abstract/AbstractText"
            ):
                part = clean_text(
                    "".join(node.itertext())
                )

                if part:
                    abstract_parts.append(part)

            abstract = " ".join(
                abstract_parts
            )

            journal = clean_text(
                article.findtext(
                    ".//Journal/Title",
                    default="",
                )
            )

            year = clean_text(
                article.findtext(
                    ".//PubDate/Year",
                    default="",
                )
            )

            if not year:
                year = clean_text(
                    article.findtext(
                        ".//PubDate/MedlineDate",
                        default="",
                    )
                )

            if not title:
                continue

            content_parts = [
                f"Title: {title}",
            ]

            if journal:
                content_parts.append(
                    f"Journal: {journal}"
                )

            if year:
                content_parts.append(
                    f"Publication date: {year}"
                )

            if pmid:
                content_parts.append(
                    f"PMID: {pmid}"
                )

            if abstract:
                content_parts.append(
                    f"Abstract: {abstract}"
                )

            content = " ".join(
                content_parts
            )

            url = (
                f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                if pmid
                else "https://pubmed.ncbi.nlm.nih.gov/"
            )

            doc = _doc(
                content=content,
                source="PubMed",
                url=url,
            )

            doc["article_title"] = title
            doc["pmid"] = pmid
            doc["match_method"] = "pubmed_search"

            docs.append(doc)

        # Rank PubMed results by direct overlap with
        # the user's medical concepts.
        query_lower = clean_text(query).lower()

        important_terms = {
            token
            for token in re.findall(
                r"[a-z0-9]+",
                query_lower,
            )
            if len(token) > 3
            and token not in {
                "what",
                "with",
                "references",
                "reference",
                "article",
                "articles",
                "study",
                "studies",
                "research",
                "subject",
                "question",
            }
        }

        def pubmed_score(doc: dict[str, Any]) -> int:
            title = clean_text(
                doc.get("article_title", "")
            ).lower()

            content = clean_text(
                doc.get("content", "")
            ).lower()

            score = 0

            for term in important_terms:
                if term in title:
                    score += 4
                elif term in content:
                    score += 1

            return score

        docs.sort(
            key=pubmed_score,
            reverse=True,
        )

        return docs[:limit]

    except Exception as exc:
        print(
            "PUBMED RETRIEVAL ERROR:",
            type(exc).__name__,
            str(exc),
        )
        return []

def retrieve_trusted_medical_web(
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Retrieve evidence from multiple trusted medical web sources,
    combine the results, remove duplicates, and rerank them
    against the user's query.
    """

    ndc = extract_ndc(query)
    drug_hints = extract_drug_hints(query)
    dosage_form = extract_dosage_form(query)

    candidate_docs: list[dict[str, Any]] = []

    # -----------------------------------------------------
    # Drug/product question
    # -----------------------------------------------------
    if ndc or drug_hints:
        if ndc:
            drug_queries = [query]
        else:
            drug_queries = [
                f"Drug: {drug}\nQuestion: {query}"
                for drug in drug_hints
            ]

        max_workers = min(
            7,
            (2 * len(drug_queries)) + 1,
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            dailymed_futures = [
                executor.submit(
                    search_dailymed,
                    query=drug_query,
                    limit=5,
                )
                for drug_query in drug_queries
            ]

            openfda_futures = [
                executor.submit(
                    search_openfda,
                    query=drug_query,
                    limit=5,
                )
                for drug_query in drug_queries
            ]

            pubmed_future = executor.submit(
                search_pubmed,
                query=query,
                limit=5,
            )

            daily_docs = [
                doc
                for future in dailymed_futures
                for doc in future.result()
            ]

            fda_docs = [
                doc
                for future in openfda_futures
                for doc in future.result()
            ]

            pubmed_docs = pubmed_future.result()

        if dosage_form:
            filtered_fda = [
                doc
                for doc in fda_docs
                if matches_dosage_form(
                    title="",
                    content=doc.get("content", ""),
                    requested_form=dosage_form,
                )
            ]

            if filtered_fda:
                fda_docs = filtered_fda

        candidate_docs.extend(daily_docs)
        candidate_docs.extend(fda_docs)
        candidate_docs.extend(pubmed_docs)

    # -----------------------------------------------------
    # Research/reference question
    # -----------------------------------------------------
    elif is_reference_question(query):
        candidate_docs = search_pubmed(
            query=query,
            limit=max(top_k, 5),
        )

    # -----------------------------------------------------
    # General medical question
    # -----------------------------------------------------
    else:
        candidate_limit = max(10, top_k * 3)

        with ThreadPoolExecutor(max_workers=2) as executor:
            medline_future = executor.submit(
                search_medlineplus,
                query=query,
                limit=candidate_limit,
            )

            pubmed_future = executor.submit(
                search_pubmed,
                query=query,
                limit=candidate_limit,
            )

            medline_docs = medline_future.result()
            pubmed_docs = pubmed_future.result()

        candidate_docs.extend(medline_docs)
        candidate_docs.extend(pubmed_docs)

    # -----------------------------------------------------
    # Remove duplicate documents
    # -----------------------------------------------------
    unique_docs = []
    seen = set()

    for doc in candidate_docs:
        key = (
            clean_text(doc.get("document_url", "")),
            clean_text(doc.get("content", "")),
        )

        if key in seen:
            continue

        seen.add(key)
        unique_docs.append(doc)

    if not unique_docs:
        return []


    import os

    if os.getenv("USE_RERANKER", "true").strip().lower() == "false":
        return unique_docs[:top_k]
    # -----------------------------------------------------
    # Rerank all trusted web evidence using ColBERT
    # -----------------------------------------------------
    try:
        ranked_docs = rerank_candidates(
            query=query,
            candidates=unique_docs,
            top_k=len(unique_docs),
        )

        if not ranked_docs:
            return []

        best_score = float(
            ranked_docs[0].get(
                "final_rerank_score",
                ranked_docs[0].get("colbert_score", 0.0),
            )
        )

        if best_score <= 0:
            return ranked_docs[:top_k]

        # Keep only documents that are reasonably close
        # to the best-ranked web evidence.
        relative_cutoff = best_score * 0.80

        filtered_docs = [
            doc
            for doc in ranked_docs
            if float(
                doc.get(
                    "final_rerank_score",
                    doc.get("colbert_score", 0.0),
                )
            )
            >= relative_cutoff
        ]

        return filtered_docs[:top_k]

    except Exception as exc:
        print(
            "TRUSTED WEB RERANK ERROR:",
            type(exc).__name__,
            str(exc),
        )

        # Safe fallback if ColBERT reranking fails.
        return unique_docs[:top_k]
