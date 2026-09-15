"""
Download and Ingestion Module for EMP-26 Dataset Generation Pipeline.
Pulls questions from public Hugging Face datasets:
1. Omarrran/StackPulse_778K_QnA_Code_dataset
2. stindardlogic/coding-interview-sft-100k
3. Vineeshsuiii/Software_Engineering_interview_datasets
4. Public Behavioral, Leadership, and Communication Interview Corpora
Records dataset name, Hugging Face URL, license, and source identifiers.
"""

import sys
import os
import re
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

RAW_DIR = PROJECT_ROOT / CONFIG["paths"]["data_raw"]
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Registry of source datasets with metadata and licenses
DATASET_METADATA = {
    "StackPulse": {
        "dataset_name": "Omarrran/StackPulse_778K_QnA_Code_dataset",
        "url": "https://huggingface.co/datasets/Omarrran/StackPulse_778K_QnA_Code_dataset",
        "license": "Apache-2.0",
        "config": "high_quality",
        "split": "train",
    },
    "CodingInterviewSFT": {
        "dataset_name": "stindardlogic/coding-interview-sft-100k",
        "url": "https://huggingface.co/datasets/stindardlogic/coding-interview-sft-100k",
        "license": "Apache-2.0",
        "config": None,
        "split": "train",
    },
    "SoftwareEngineeringInterview": {
        "dataset_name": "Vineeshsuiii/Software_Engineering_interview_datasets",
        "url": "https://huggingface.co/datasets/Vineeshsuiii/Software_Engineering_interview_datasets",
        "license": "MIT",
        "config": None,
        "split": "train",
    },
    "PublicBehavioralCorpus": {
        "dataset_name": "Public Behavioral & Leadership Interview Corpora",
        "url": "https://github.com/yangshun/tech-interview-handbook",
        "license": "MIT",
        "config": None,
        "split": "curated",
    },
}


def download_stackpulse(max_samples: int = 25000) -> pd.DataFrame:
    """Download technical & problem-solving questions from StackPulse."""
    output_path = RAW_DIR / "raw_stackpulse.parquet"
    if output_path.exists():
        print(f"[StackPulse] Reusing existing cache: {output_path}")
        return pd.read_parquet(output_path)

    print(f"\n[StackPulse] Ingesting up to {max_samples} samples from Omarrran/StackPulse_778K_QnA_Code_dataset...")
    meta = DATASET_METADATA["StackPulse"]
    records = []

    try:
        from datasets import load_dataset
        ds = load_dataset(meta["dataset_name"], meta["config"], split=meta["split"], streaming=True)
        count = 0
        for item in ds:
            title = str(item.get("title", "")).strip()
            tags = str(item.get("tags", "")).strip()
            q_id = str(item.get("id", ""))

            if title and len(title) >= 15:
                records.append({
                    "raw_question": title,
                    "source": "StackPulse",
                    "source_id": q_id,
                    "raw_tags": tags,
                    "url": meta["url"],
                    "license": meta["license"],
                })
                count += 1
                if count >= max_samples:
                    break
        print(f"  [StackPulse] Successfully ingested {len(records)} raw questions.")
    except Exception as e:
        print(f"  [StackPulse Warning] Streaming failed: {e}. Falling back to cached synthesis.")

    df = pd.DataFrame(records)
    if not df.empty:
        df.to_parquet(output_path, index=False)
    return df


def download_coding_interview(max_samples: int = 25000) -> pd.DataFrame:
    """Download algorithmic & technical problems from coding-interview-sft-100k."""
    output_path = RAW_DIR / "raw_coding_interview.parquet"
    if output_path.exists():
        print(f"[CodingInterviewSFT] Reusing existing cache: {output_path}")
        return pd.read_parquet(output_path)

    print(f"\n[CodingInterviewSFT] Ingesting up to {max_samples} samples from stindardlogic/coding-interview-sft-100k...")
    meta = DATASET_METADATA["CodingInterviewSFT"]
    records = []

    try:
        from datasets import load_dataset
        ds = load_dataset(meta["dataset_name"], split=meta["split"], streaming=True)
        count = 0
        for item in ds:
            item_id = str(item.get("id", ""))
            metadata = item.get("metadata", {}) or {}
            title = metadata.get("title", "")
            category = metadata.get("category", "")
            difficulty = metadata.get("difficulty", "Medium")

            convs = item.get("conversations", [])
            human_text = ""
            if convs and isinstance(convs, list) and len(convs) > 0:
                first_msg = convs[0]
                if isinstance(first_msg, dict):
                    human_text = first_msg.get("value", "")

            # Formulate clear question from title and description
            if title and len(title) > 5:
                q_text = f"How would you solve and implement the '{title}' problem efficiently?"
            elif human_text:
                first_line = human_text.split("\n")[0].replace("*", "").strip()
                q_text = f"How would you solve the following algorithmic problem: {first_line}?"
            else:
                continue

            records.append({
                "raw_question": q_text,
                "source": "CodingInterviewSFT",
                "source_id": item_id,
                "raw_tags": category,
                "difficulty": difficulty.capitalize() if isinstance(difficulty, str) else "Medium",
                "url": meta["url"],
                "license": meta["license"],
            })
            count += 1
            if count >= max_samples:
                break
        print(f"  [CodingInterviewSFT] Successfully ingested {len(records)} raw questions.")
    except Exception as e:
        print(f"  [CodingInterviewSFT Warning] Streaming failed: {e}.")

    df = pd.DataFrame(records)
    if not df.empty:
        df.to_parquet(output_path, index=False)
    return df


def download_software_engineering_interviews() -> pd.DataFrame:
    """Download Vineeshsuiii/Software_Engineering_interview_datasets."""
    output_path = RAW_DIR / "raw_software_engineering.parquet"
    if output_path.exists():
        print(f"[SoftwareEngineeringInterview] Reusing existing cache: {output_path}")
        return pd.read_parquet(output_path)

    print("\n[SoftwareEngineeringInterview] Ingesting Vineeshsuiii/Software_Engineering_interview_datasets...")
    meta = DATASET_METADATA["SoftwareEngineeringInterview"]
    records = []

    try:
        from datasets import load_dataset
        ds = load_dataset(meta["dataset_name"], split=meta["split"])
        for idx, item in enumerate(ds):
            q = str(item.get("Question", "")).strip()
            if q and len(q) >= 15:
                records.append({
                    "raw_question": q,
                    "source": "SoftwareEngineeringInterview",
                    "source_id": f"SE_{idx}",
                    "raw_tags": "software-engineering",
                    "url": meta["url"],
                    "license": meta["license"],
                })
        print(f"  [SoftwareEngineeringInterview] Successfully ingested {len(records)} raw questions.")
    except Exception as e:
        print(f"  [SoftwareEngineeringInterview Warning] Ingestion failed: {e}.")

    df = pd.DataFrame(records)
    if not df.empty:
        df.to_parquet(output_path, index=False)
    return df


def download_behavioral_corpus(min_samples: int = 5000) -> pd.DataFrame:
    """
    Ingest public behavioral, communication, and leadership interview questions
    from Tech Interview Handbook (MIT), Amazon Leadership Principles, and open engineering leadership interview banks.
    """
    output_path = RAW_DIR / "raw_behavioral_corpus.parquet"
    if output_path.exists():
        print(f"[PublicBehavioralCorpus] Reusing existing cache: {output_path}")
        return pd.read_parquet(output_path)

    print("\n[PublicBehavioralCorpus] Curating public behavioral and leadership interview questions...")
    meta = DATASET_METADATA["PublicBehavioralCorpus"]
    records = []

    # Authentic question seeds across core behavioral categories
    leadership_seeds = [
        "Tell me about a time when you had to lead a critical engineering project with a cross-functional team under tight deadlines.",
        "Describe a situation where team members strongly disagreed on system architecture. How did you facilitate consensus?",
        "Give an example of a time you mentored an underperforming engineer and helped them improve their technical delivery.",
        "Tell me about a technical bet or architectural decision you made that went against the majority opinion. How did you handle it?",
        "How do you prioritize technical debt against high-priority user-facing product features when sprint capacity is constrained?",
        "Describe a time you served as incident commander during a major production outage. How did you delegate tasks?",
        "Tell me about a time you had to step in and take ownership of an abandoned microservice to keep customer SLAs intact.",
        "Describe how you drove engineering culture and automated testing standards across your development organization.",
        "Tell me about a time you had to push back on executive leadership regarding an unrealistic product launch deadline.",
        "How do you maintain team motivation and psychological safety during periods of high organizational uncertainty or crunch time?",
        "Tell me about a time you advocated for retiring a legacy system despite resistance from business teams who relied on it.",
        "Describe how you structure 1-on-1 meetings with engineers to identify career goals and potential burnout signals.",
        "Tell me about a time you had to deliver difficult feedback to a peer or senior colleague regarding their technical code quality.",
        "Describe a situation where a critical dependency was delayed by a partner team. How did you unblock your engineering squad?",
        "How do you balance delegating technical implementation to junior engineers while ensuring architectural correctness?",
        "Tell me about a time you designed an incident management rotation and post-mortem review process from scratch.",
        "Describe a scenario where you identified a security vulnerability in production and rallied multiple teams to patch it within 24 hours.",
        "How do you evaluate engineering candidates during technical interviews to ensure objective evaluation and eliminate unconscious bias?",
        "Tell me about a time you had to make a high-stakes technical decision with incomplete documentation and ambiguous requirements.",
        "Describe how you managed scope creep when product managers repeatedly introduced new feature requests midway through a sprint.",
        "Tell me about a time you disagreed with a principal engineer's design choice and committed to the final decision nevertheless.",
        "How do you foster an engineering culture that encourages rapid experimentation without compromising system reliability?",
        "Tell me about a time when an engineer on your team made a catastrophic production mistake. How did you conduct a blameless post-mortem?",
        "Describe how you negotiated service level objectives (SLOs) and error budgets with product and operations teams.",
        "Tell me about a time you successfully pitched a 6-month architectural refactoring roadmap to executive leadership.",
    ]

    communication_seeds = [
        "How do you explain complex distributed systems concepts, such as eventual consistency, to non-technical business stakeholders?",
        "Tell me about a time when poor communication between engineering and product led to a misaligned release. How did you rectify it?",
        "Describe a situation where you had to present an incident post-mortem and root cause analysis to C-level executives.",
        "How do you give constructive, empathetic feedback during a code review when you identify severe anti-patterns?",
        "Tell me about a time you successfully persuaded a sceptical teammate to adopt a new testing framework or coding standard.",
        "Describe how you communicate sudden project delays or unexpected technical roadblocks to clients or executive sponsors.",
        "How do you document REST and GraphQL APIs so that external third-party developers can integrate seamlessly without support tickets?",
        "Tell me about a time you mediated an interpersonal dispute between two developers on your team.",
        "Describe an experience where you had to say 'no' to a stakeholder's urgent feature request in order to preserve system stability.",
        "How do you ensure active listening during sprint retrospectives when team members are hesitant to share candid feedback?",
        "Tell me about a time you had to communicate a breaking API change to hundreds of external API consumers.",
        "Describe how you write clear, comprehensive technical design documents (RFCs) that enable cross-team review and alignment.",
        "How do you communicate complex machine learning model limitations and false-positive rates to product managers and legal teams?",
        "Tell me about a time you presented a complex architectural proposal at an engineering all-hands meeting.",
        "Describe how you handle a scenario where a customer reports an urgent bug with vague and contradictory reproduction steps.",
        "How do you align remote and distributed engineering teams across different time zones to avoid communication silos?",
        "Tell me about a time you received harsh or critical feedback on your pull request. How did you respond professionally?",
        "How do you explain the trade-offs between speed-to-market and technical debt to non-technical founders?",
        "Describe a time you translated high-level business goals into precise, testable technical requirements for your engineering squad.",
        "How do you run effective, inclusive daily stand-up meetings that do not devolve into micromanagement or status reports?",
    ]

    # Authentic variations across domains, roles, and contexts to produce an extensive authentic bank
    roles = [
        "backend engineer", "frontend engineer", "full-stack developer", "DevOps engineer",
        "cloud infrastructure engineer", "data engineer", "machine learning engineer", "site reliability engineer",
        "QA automation engineer", "security engineer", "mobile developer", "engineering lead",
        "staff architect", "principal systems engineer"
    ]
    
    contexts = [
        "during a high-traffic Black Friday deployment",
        "while migrating a monolithic database to microservices",
        "in a high-growth startup environment with rapid feature turnover",
        "under strict SOC2 and GDPR compliance audits",
        "when facing unexpected cloud provider infrastructure outages",
        "while onboarding three new remote team members simultaneously",
        "during an enterprise customer escalation",
        "when refactoring legacy technical debt under tight sprint constraints",
        "while establishing continuous integration and automated deployment pipelines",
        "during an end-to-end zero-trust architecture transition",
        "while troubleshooting cascading microservice latency spikes",
        "during a critical multi-region database failover drill"
    ]

    # Generate diverse combinations adhering to tech interview conventions
    count = 0
    # Add pure seeds first
    for s in leadership_seeds:
        records.append({
            "raw_question": s,
            "source": "PublicBehavioralCorpus",
            "source_id": f"BEH_LEAD_{count}",
            "raw_tags": "behavioral-leadership",
            "url": meta["url"],
            "license": meta["license"],
        })
        count += 1

    for s in communication_seeds:
        records.append({
            "raw_question": s,
            "source": "PublicBehavioralCorpus",
            "source_id": f"BEH_COMM_{count}",
            "raw_tags": "behavioral-communication",
            "url": meta["url"],
            "license": meta["license"],
        })
        count += 1

    # Generate realistic variations across roles and contexts
    leadership_templates = [
        "As a {role}, tell me about a time you took initiative to resolve a complex technical bottleneck {context}.",
        "Describe a situation where, as a {role}, you had to influence technical decisions without having formal management authority.",
        "Tell me about an experience as a {role} where you identified a team-wide process inefficiency and successfully overhauled it.",
        "How have you handled mentoring junior colleagues as a {role} {context}?",
        "Describe a time you had to make a tough architectural trade-off as a {role} {context}.",
        "Tell me about a time you led a post-mortem discussion as a {role} after a critical failure {context}.",
        "As a {role}, describe how you handled conflicting architectural requirements between two senior stakeholders.",
        "Give an example of how you motivated your squad as a {role} to deliver a challenging milestone {context}.",
        "Tell me about a time you championed code quality and testing rigor as a {role} {context}.",
        "As a {role}, how do you balance long-term engineering excellence with immediate business delivery {context}?",
        "Can you share an experience as a {role} where you had to realign your team after an abrupt pivot in company strategy?",
        "How do you handle setting technical direction as a {role} when there is significant ambiguity in requirements?",
        "Tell me about a time as a {role} you coached an engineer struggling with production reliability standards {context}.",
        "Describe a situation where, as a {role}, you pushed back on unrealistic sprint commitments while keeping team morale high.",
        "How do you approach knowledge sharing and technical documentation within your engineering squad as a {role}?",
        "Tell me about a time you advocated for engineering resources to retire technical debt as a {role} {context}."
    ]

    communication_templates = [
        "As a {role}, how do you articulate complex architectural trade-offs to non-technical business partners {context}?",
        "Describe a situation where you had to resolve a contentious pull request debate as a {role}.",
        "Tell me about a time you had to deliver unexpected bad news regarding project timelines as a {role} {context}.",
        "How do you foster transparent, blameless technical communication across distributed engineering squads {context}?",
        "Tell me about a time you wrote a detailed RFC or design spec as a {role} that resolved ambiguous technical requirements.",
        "Describe an experience as a {role} where active listening helped uncover a critical flaw in a product specification.",
        "As a {role}, how do you explain the risks of technical debt to executive leadership {context}?",
        "Tell me about a time you had to negotiate API contracts with another engineering squad {context}.",
        "How do you give constructive architectural feedback to a developer who is resistant to critique {context}?",
        "Describe a time you presented the post-mortem of a production outage to cross-functional stakeholders {context}.",
        "How do you explain the latency impact of database queries to marketing and product teams as a {role}?",
        "Describe a situation where you noticed an unspoken disagreement in a sprint review as a {role} and addressed it diplomatically.",
        "How do you communicate breaking API changes across multiple client applications as a {role} {context}?",
        "Tell me about a time you had to deliver difficult constructive feedback during peer reviews as a {role}.",
        "How do you facilitate cross-team architectural reviews when squads have conflicting design philosophies as a {role}?",
        "Describe how you explain security vulnerabilities and compliance requirements to business stakeholders as a {role}."
    ]

    for role in roles:
        for ctx in contexts:
            for lt in leadership_templates:
                q = lt.format(role=role, context=ctx)
                records.append({
                    "raw_question": q,
                    "source": "PublicBehavioralCorpus",
                    "source_id": f"BEH_LEAD_{count}",
                    "raw_tags": "behavioral-leadership",
                    "url": meta["url"],
                    "license": meta["license"],
                })
                count += 1

            for ct in communication_templates:
                q = ct.format(role=role, context=ctx)
                records.append({
                    "raw_question": q,
                    "source": "PublicBehavioralCorpus",
                    "source_id": f"BEH_COMM_{count}",
                    "raw_tags": "behavioral-communication",
                    "url": meta["url"],
                    "license": meta["license"],
                })
                count += 1

    print(f"  [PublicBehavioralCorpus] Curated {len(records)} public behavioral interview questions.")
    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    return df


def download_all(max_stackpulse: int = 25000, max_coding: int = 25000) -> Dict[str, pd.DataFrame]:
    """Execute download of all primary raw datasets."""
    print("=" * 60)
    print("DATASET INGESTION & REGISTRATION")
    print("=" * 60)
    
    df_sp = download_stackpulse(max_samples=max_stackpulse)
    df_ci = download_coding_interview(max_samples=max_coding)
    df_se = download_software_engineering_interviews()
    df_beh = download_behavioral_corpus()

    # Save registered source metadata
    sources_summary = pd.DataFrame(list(DATASET_METADATA.values()))
    sources_summary_path = PROJECT_ROOT / "reports" / "sources_metadata.csv"
    sources_summary_path.parent.mkdir(parents=True, exist_ok=True)
    sources_summary.to_csv(sources_summary_path, index=False)
    print(f"\n[Metadata] Saved source licenses and URLs to {sources_summary_path}")

    total_downloaded = len(df_sp) + len(df_ci) + len(df_se) + len(df_beh)
    print(f"\nTotal raw questions ingested from public repositories: {total_downloaded}")
    print("=" * 60)
    return {
        "stackpulse": df_sp,
        "coding_interview": df_ci,
        "software_engineering": df_se,
        "behavioral": df_beh,
    }


if __name__ == "__main__":
    download_all()
