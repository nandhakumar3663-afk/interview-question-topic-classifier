"""
Curated and synthetic dataset generator for EMP-26 Interview Question Topic Classifier.
Generates 3,500+ diverse, realistic, industry-standard interview questions across
all 5 target categories with balanced representations, then cleans, deduplicates,
and generates stratified 70/15/15 train/validation/test splits.
"""

import sys
import random
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure src can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils import (
    TOPIC_CATEGORIES,
    RAW_QUESTIONS_PATH,
    LABELED_QUESTIONS_PATH,
    TRAIN_SPLIT_PATH,
    VAL_SPLIT_PATH,
    TEST_SPLIT_PATH,
    ensure_directories,
)
from src.preprocessing import clean_dataset, clean_text

# Seed for reproducibility
random.seed(42)

# ==========================================
# Domain Dictionaries for Balanced Generation
# ==========================================

TECHNICAL_TEMPLATES = [
    "What is the difference between {tech_a} and {tech_b}?",
    "How does {tech_concept} work under the hood in modern distributed systems?",
    "Can you explain the trade-offs of using {tech_concept} in production?",
    "What are the best practices for implementing {tech_concept} in a high-scale service?",
    "How would you design a system that handles {scale_metric} using {tech_concept}?",
    "Explain how {tech_concept} impacts latency, throughput, and fault tolerance.",
    "What is the time and space complexity of {algorithm} and how can it be optimized?",
    "How do you handle data consistency and isolation levels in {db_tech}?",
    "What are the internal mechanics of {runtime_feature} in {language}?",
    "Explain the principles of {architectural_pattern} and when you would avoid it.",
    "How do you resolve {concurrency_issue} in multithreaded applications?",
    "What happens from a networking perspective when {network_event} occurs?",
    "How do indexes in {db_tech} improve query performance and what are the overhead costs?",
    "Can you explain the difference between {paradigm_a} and {paradigm_b} in software engineering?",
    "How does {cache_tech} handle cache invalidation, cache stampede, and eviction policies?",
    "Explain the CAP theorem and how {tech_concept} chooses between consistency and availability.",
    "How does garbage collection work in {language} and how can you minimize stop-the-world pauses?",
    "What is the difference between synchronous and asynchronous processing in {tech_concept}?",
    "How do you design an idempotent API endpoint for {api_use_case}?",
    "Explain how {security_concept} protects against common vulnerabilities in modern web applications.",
]

TECHNICAL_VARS = {
    "tech_a": ["TCP", "REST", "SQL databases", "monolithic architecture", "processes", "optimistic locking", "HTTP/2", "B-tree index", "Kafka", "symmetric encryption"],
    "tech_b": ["UDP", "GraphQL", "NoSQL databases", "microservices", "threads", "pessimistic locking", "HTTP/3", "Hash index", "RabbitMQ", "asymmetric encryption"],
    "tech_concept": ["database sharding", "event-driven architecture", "distributed transactions (2PC)", "read-write replicas", "connection pooling", "circuit breaker pattern", "rate limiting", "consistent hashing", "WAL (write-ahead log)", "CQRS pattern", "message broker partitioning", "memory-mapped files"],
    "scale_metric": ["100,000 requests per second", "real-time analytics over petabytes of data", "low-latency financial transactions under 5ms", "geo-distributed user sessions with instant failover", "millions of concurrent WebSocket connections"],
    "algorithm": ["QuickSort vs MergeSort", "Dijkstra's shortest path algorithm", "Binary Search on a rotated sorted array", "Topological Sort for DAG dependencies", "LRU Cache eviction with O(1) operations", "Trie prefix matching", "K-way merge on distributed streams"],
    "db_tech": ["PostgreSQL", "MySQL InnoDB", "MongoDB", "Cassandra", "DynamoDB", "Redis", "Elasticsearch"],
    "runtime_feature": ["memory allocation and heap management", "virtual method dispatch and vtables", "async/await event loop scheduling", "goroutines vs OS threads", "reflection and dynamic type inspection", "GIL (Global Interpreter Lock)"],
    "language": ["Java", "Python", "Go", "C++", "C#", "Rust"],
    "architectural_pattern": ["Domain-Driven Design (DDD)", "Clean Architecture", "Hexagonal Architecture", "Event Sourcing", "Micro-frontends", "Service Mesh"],
    "concurrency_issue": ["race conditions and deadlocks", "thread starvation and livelocks", "shared memory mutation without mutexes", "the ABA problem in lock-free data structures"],
    "network_event": ["a user types a URL into a browser until the page renders", "a TLS 1.3 handshake is established", "a packet travels across NAT gateways and routers", "a DNS recursive resolution query is executed"],
    "paradigm_a": ["object-oriented programming", "declarative programming", "stateful services", "strong consistency"],
    "paradigm_b": ["functional programming", "imperative programming", "stateless microservices", "eventual consistency"],
    "cache_tech": ["Redis", "Memcached", "CDN edge caching", "in-memory Guava cache"],
    "api_use_case": ["credit card billing payments", "user order placement checkout", "webhook notifications delivery", "inventory reservation systems"],
    "security_concept": ["OAuth2 with PKCE", "JWT token signing and refresh rotation", "CORS policy enforcement", "content security policies (CSP)"],
}

COMMUNICATION_TEMPLATES = [
    "How do you explain {complex_concept} to {non_tech_audience}?",
    "Describe a time when you had to convince {stakeholder} to prioritize {engineering_need}.",
    "How do you handle situations where a {stakeholder} demands a delivery deadline you believe is unrealistic?",
    "Tell me about a difficult conversation you had with {counterpart} and how you reached mutual agreement.",
    "How do you communicate a critical production outage to {stakeholder} during active incident response?",
    "Describe how you present technical trade-offs to {non_tech_audience} when choosing between {option_a} and {option_b}.",
    "How do you ensure requirements and technical architecture are clearly understood across a distributed team?",
    "Can you share an experience where miscommunication led to a bug or delay, and how you corrected it?",
    "How do you deliver constructive, critical code review feedback without offending {counterpart}?",
    "What is your approach to writing clear technical documentation, RFCs, and architectural decision records?",
    "How do you handle disagreements during an architectural design meeting when consensus cannot be reached?",
    "Describe a scenario where you negotiated the scope of a feature release with {stakeholder}.",
    "How do you facilitate discussions between teams that have conflicting technical priorities or API requirements?",
    "What strategies do you use for active listening when a colleague expresses frustration with your proposal?",
    "How do you communicate the long-term cost of technical debt to executive leadership who only want new features?",
    "Tell me about a time you had to deliver bad news regarding project delays to {stakeholder}.",
    "How do you articulate complex system constraints during cross-functional sprint planning sessions?",
    "Describe how you run a blameless post-mortem so the team communicates openly about mistakes.",
    "How do you tailor your communication style when speaking with junior engineers versus C-level executives?",
    "Tell me about a time you successfully persuaded a skeptical teammate to adopt a new testing methodology.",
]

COMMUNICATION_VARS = {
    "complex_concept": ["database sharding and eventual consistency", "why refactoring technical debt prevents system crashes", "the difference between latency and throughput bottlenecks", "why upgrading outdated runtime versions is critical for security", "why distributed consensus algorithms cannot guarantee zero latency"],
    "non_tech_audience": ["non-technical business executives", "product managers without coding background", "sales and marketing stakeholders", "customer support specialists", "client account managers"],
    "stakeholder": ["the VP of Product", "an impatient product owner", "the enterprise customer success director", "the Chief Operating Officer", "a non-technical client sponsor"],
    "engineering_need": ["a comprehensive architectural refactor", "migrating legacy monolithic databases to microservices", "allocating 30% of sprint capacity to test automation", "pausing new feature rollouts to fix critical security vulnerabilities"],
    "counterpart": ["a peer software engineer with strong opinions", "a cross-functional product manager", "a strict QA lead", "an external vendor technical contact", "a legacy system maintainer"],
    "option_a": ["faster time-to-market with accumulated tech debt", "a managed cloud proprietary service", "building a customized in-house solution"],
    "option_b": ["a resilient scalable architecture with higher initial development cost", "an open-source self-hosted cluster", "purchasing an off-the-shelf third-party integration"],
}

PROBLEM_SOLVING_TEMPLATES = [
    "Tell me about a time when you diagnosed and fixed {difficult_incident} in production.",
    "How do you approach troubleshooting an intermittent issue that {intermittent_behavior}?",
    "Walk me through your step-by-step debugging process when {symptom} occurs on a mission-critical service.",
    "Describe a complex technical problem where you had very limited log data and monitoring. How did you isolate root cause?",
    "Can you share an example of a time when your initial hypothesis about a severe bug turned out to be wrong?",
    "How do you identify whether a performance slowdown is caused by {cause_a} or {cause_b}?",
    "Describe a situation where a third-party dependency failed unexpectedly in production. How did you resolve the outage?",
    "How do you troubleshoot high CPU utilization and thread starvation when memory metrics appear completely normal?",
    "Tell me about a time when you resolved a catastrophic database deadlock under peak traffic.",
    "How do you triage and prioritize five simultaneous critical production alerts during an on-call shift?",
    "Walk me through how you discovered and patched a silent data corruption bug in an asynchronous processing pipeline.",
    "Describe an incident where a faulty deployment caused widespread 500 server errors. What was your immediate mitigation?",
    "How do you debug an application that runs perfectly in local staging but fails intermittently in Kubernetes production?",
    "Tell me about a time you used profiling tools (e.g., Flamegraphs, pprof, VisualVM) to resolve a massive latency spike.",
    "How do you diagnose memory leaks in long-running containerized microservices without restarting the cluster?",
    "Describe a time when you had to make a high-stakes engineering trade-off under intense time pressure.",
    "How do you determine the root cause of network packet loss and TCP retransmission timeouts between microservices?",
    "Walk me through how you solved a stubborn distributed race condition that only occurred once every thousand requests.",
    "Tell me about a time you investigated a critical security breach or SQL injection attempt. How did you contain the blast radius?",
    "What methodology do you apply when decomposing a messy, undocumented legacy system bug?",
    "How would you troubleshoot sudden memory exhaustion caused by {difficult_incident}?",
    "Walk me through how you would isolate an issue when {symptom} happens right after a major framework upgrade.",
    "Describe a challenging post-incident review where the root cause was traced back to {cause_a} rather than {cause_b}.",
    "Tell me about a situation where an automated rollback failed and you had to resolve {difficult_incident} manually.",
    "How do you investigate intermittent latency regressions when {cause_b} is suspected?",
]

PROBLEM_SOLVING_VARS = {
    "difficult_incident": [
        "a critical memory leak causing cascading OOM kills",
        "an exponential spike in 504 Gateway Timeouts",
        "a connection pool exhaustion during a flash sale",
        "a silent queue backlog causing millions of dropped events",
        "an unindexed query locking the primary production database",
        "a Redis cluster split-brain event causing state inconsistency",
        "an unexpected thread deadlock freezing background queue workers",
        "a runaway recursive background job consuming all available CPU",
        "an avalanche of unhandled exceptions crashing microservice pods",
        "a sudden disk I/O saturation freezing asynchronous write operations",
    ],
    "intermittent_behavior": [
        "only reproduces under high concurrent traffic",
        "fails unpredictably once every several thousand requests",
        "disappears immediately whenever verbose debug logging is enabled",
        "only manifests across specific geographical cloud regions",
        "triggers only after a service has been running continuously for several days",
        "occurs only during automated batch database backups",
        "happens exclusively on mobile clients with unstable network connections",
    ],
    "symptom": [
        "the API response time degrades from 50ms to 12 seconds",
        "worker nodes abruptly crash without emitting a stack trace",
        "database replication lag surges past 30 minutes",
        "clients receive partial and corrupted payload responses",
        "the HTTP request error rate spikes to 25%",
        "Kafka consumer lag continuously grows without consumer rebalancing",
        "memory consumption steadily climbs linearly until container restarts",
    ],
    "cause_a": [
        "database connection pooling saturation",
        "CPU throttling by Kubernetes CFS quota",
        "DNS query resolution latency",
        "garbage collection pause times",
        "thread lock contention on shared resource caches",
        "unbounded in-memory queue buffering",
    ],
    "cause_b": [
        "application thread lock contention",
        "underlying cloud network packet loss",
        "downstream third-party rate limits",
        "unoptimized disk I/O swapping",
        "microservice circuit breaker tripping prematurely",
        "TCP socket file descriptor leaks",
    ],
}

LEADERSHIP_TEMPLATES = [
    "Describe a time when you mentored a {mentee_type} who was struggling with {struggle_area}.",
    "How do you handle a situation where two senior engineers on your team strongly disagree on {disagreement_topic}?",
    "Tell me about an initiative you led from conception to delivery across multiple cross-functional teams.",
    "How do you establish engineering standards, code quality benchmarks, and automated testing rigor across a team?",
    "Describe a scenario where you had to lead a project with ambiguous requirements and tight organizational deadlines.",
    "How do you keep team morale and velocity high during intense crunch periods or corporate restructuring?",
    "Tell me about a time you noticed an engineer feeling burned out or disengaged. What actions did you take?",
    "How do you balance delegating important tasks to junior developers while maintaining high delivery quality?",
    "Describe a time when you had to manage an underperforming team member. How did you handle the situation empathetically?",
    "How do you drive consensus on adopting {new_tech_paradigm} when there is substantial resistance from legacy developers?",
    "Tell me about a time you championed a cultural change or diversity initiative within your engineering department.",
    "How do you run effective 1-on-1 meetings that support the long-term career growth of your direct reports?",
    "Describe a situation where executive management proposed an unreasonable technical mandate. How did you push back?",
    "How do you foster a psychological safety culture where developers feel safe admitting mistakes and proposing bold ideas?",
    "Tell me about a time you successfully onboarded multiple new engineers while maintaining ongoing sprint commitments.",
    "How do you recognize and develop emerging leadership qualities in engineers on your team?",
    "Describe how you prioritize technical debt against business feature roadmap requests at an organizational level.",
    "Tell me about a time when a critical project you were leading was failing. How did you turn it around?",
    "How do you navigate team member promotions, performance evaluations, and compensation reviews fairly?",
    "What is your philosophy on servant leadership versus top-down command in high-performing engineering teams?",
    "How do you motivate a cross-functional squad to align on {new_tech_paradigm} without causing delivery friction?",
    "Tell me about a time you stepped up to resolve an organizational deadlock involving {disagreement_topic}.",
    "Describe how you supported a {mentee_type} in transitioning to higher-impact architectural responsibilities.",
    "How do you balance technical vision with day-to-day tactical execution as a team lead?",
    "Tell me about a time you had to deliver difficult feedback about {struggle_area} while maintaining trust.",
]

LEADERSHIP_VARS = {
    "mentee_type": [
        "junior software engineer",
        "mid-level engineer aiming for senior promotion",
        "intern with minimal real-world experience",
        "career switcher unfamiliar with production pipelines",
        "newly hired backend developer adjusting to company codebase",
        "talented engineer struggling with imposter syndrome",
    ],
    "struggle_area": [
        "code quality and test design",
        "time management and estimation accuracy",
        "confidence in contributing to architectural design reviews",
        "handling ambiguous technical specifications",
        "proactive communication during blockers",
        "understanding distributed systems trade-offs",
        "writing thorough documentation and unit tests",
    ],
    "disagreement_topic": [
        "migrating to microservices versus modular monolith",
        "choosing between Go and Rust for a new microservice",
        "enforcing strict functional programming paradigms",
        "authorizing third-party cloud vendor lock-in",
        "introducing GraphQL over RESTful API architecture",
        "enforcing 90% code coverage requirements on pull requests",
        "decoupling legacy monolithic database schemas",
    ],
    "new_tech_paradigm": [
        "automated CI/CD trunk-based development",
        "strict test-driven development (TDD)",
        "mandatory peer code reviews and static security scanning",
        "moving to container orchestration and GitOps",
        "implementing observability and distributed tracing standards",
        "adopting asynchronous event-driven architectures",
    ],
}

ROLE_SPECIFIC_TEMPLATES = [
    "In {domain_field}, how do you implement {domain_concept} to ensure high performance?",
    "What are the key trade-offs between {tech_alt_a} and {tech_alt_b} when working on {domain_task}?",
    "How do you configure and optimize {tool_or_framework} for enterprise production workloads?",
    "Describe how you design a resilient pipeline for {pipeline_type} in modern cloud infrastructure.",
    "What strategies do you use for {specialized_activity} in {domain_field}?",
    "How do you ensure accessibility, cross-browser compatibility, and WCAG compliance in {frontend_work}?",
    "Explain how you manage state and avoid unnecessary re-renders in {ui_framework} applications.",
    "How do you write reliable, maintainable end-to-end integration tests using {test_tool} without creating flaky tests?",
    "Describe how you manage Terraform state files, lock mechanisms, and module hierarchies in multi-environment setups.",
    "How do you architect an Apache Spark streaming job to handle out-of-order event timestamps and late data arriving?",
    "What best practices do you follow for building reusable design system components in {ui_framework}?",
    "How do you implement OAuth2 Authorization Code Flow with PKCE in a Single Page Application (SPA)?",
    "Describe the configuration of Kubernetes Horizontal Pod Autoscalers (HPA) based on custom Prometheus metrics.",
    "How do you manage database migrations safely with zero-downtime using tools like Flyway or Liquibase in {domain_field}?",
    "What is your approach to setting up distributed tracing with OpenTelemetry and Jaeger across microservices?",
    "How do you secure a CI/CD pipeline against supply chain attacks, compromised packages, and secret leakage?",
    "Explain how you design a data lakehouse architecture using Delta Lake or Apache Iceberg.",
    "How do you profile and eliminate bundle bloat, tree-shaking failures, and excessive CSS in modern web applications?",
    "What is the difference between Blue-Green deployments and Canary deployments in Kubernetes?",
    "How do you design a comprehensive regression testing matrix for a critical payment gateway service?",
]

ROLE_SPECIFIC_VARS = {
    "domain_field": ["Frontend Engineering", "DevOps & Cloud SRE", "Data Engineering", "QA Automation", "Cybersecurity Engineering"],
    "domain_concept": ["virtualized list rendering for 100k items", "automated canary rollouts with ArgoCD", "idempotent ELT data pipelines in Snowflake", "contract testing between microservices with Pact", "zero-trust identity-aware network segmentation"],
    "tech_alt_a": ["React Server Components", "Docker Compose", "Apache Kafka", "Cypress", "AWS Lambda serverless"],
    "tech_alt_b": ["Client-Side Hydration", "Kubernetes Pods", "Apache Flink", "Playwright", "Kubernetes long-running deployments"],
    "domain_task": ["frontend state synchronization", "cloud container orchestration", "real-time clickstream processing", "end-to-end regression validation", "microservice communication"],
    "tool_or_framework": ["Kubernetes ingress controllers", "Apache Airflow DAG schedulers", "Next.js SSR caching layers", "Prometheus and Grafana dashboards", "HashiCorp Vault for secrets"],
    "pipeline_type": ["continuous deployment with automatic rollback", "real-time financial transaction ingestion", "automated test suite execution across parallel runners", "zero-downtime database schema migration"],
    "specialized_activity": ["optimizing Core Web Vitals (LCP, INP, CLS)", "hardening Linux kernel parameters against DDoS", "partition pruning and shuffling optimization", "mocking flaky third-party payment endpoints"],
    "frontend_work": ["complex single-page applications", "enterprise dashboard portals", "e-commerce checkout funnels", "mobile-responsive web applications"],
    "ui_framework": ["React", "Vue 3", "Angular", "Svelte"],
    "test_tool": ["Playwright", "Cypress", "Selenium Grid", "Jest and React Testing Library"],
}


# ==========================================
# Question Generation Core Functions
# ==========================================

def fill_template(template: str, vars_dict: dict) -> str:
    """Safely fill a template by picking random items for variables present in template."""
    chosen_vars = {}
    for var_name, var_list in vars_dict.items():
        if f"{{{var_name}}}" in template:
            chosen_vars[var_name] = random.choice(var_list)
    return template.format(**chosen_vars)


def generate_category_questions(category: str, target_count: int) -> list:
    """Generate target_count unique questions for a given topic category."""
    questions = set()
    
    if category == "Technical Knowledge":
        templates = TECHNICAL_TEMPLATES
        vars_dict = TECHNICAL_VARS
    elif category == "Communication":
        templates = COMMUNICATION_TEMPLATES
        vars_dict = COMMUNICATION_VARS
    elif category == "Problem Solving":
        templates = PROBLEM_SOLVING_TEMPLATES
        vars_dict = PROBLEM_SOLVING_VARS
    elif category == "Leadership":
        templates = LEADERSHIP_TEMPLATES
        vars_dict = LEADERSHIP_VARS
    elif category == "Role-Specific Skills":
        templates = ROLE_SPECIFIC_TEMPLATES
        vars_dict = ROLE_SPECIFIC_VARS
    else:
        raise ValueError(f"Unknown category: {category}")

    # Prefixes / variations to introduce natural syntax variations
    variations = [
        "",
        "In your previous projects, ",
        "Could you walk me through: ",
        "From an architectural standpoint, ",
        "In an enterprise environment, ",
        "During a technical review, ",
        "Can you explain: ",
        "Hypothetically, ",
        "As a senior engineer, ",
        "From your experience, ",
    ]

    attempts = 0
    max_attempts = target_count * 50

    while len(questions) < target_count and attempts < max_attempts:
        attempts += 1
        template = random.choice(templates)
        q = fill_template(template, vars_dict)
        
        # Optionally add variation prefix
        var_prefix = random.choice(variations)
        if var_prefix and not q.lower().startswith(("what", "how", "tell", "describe", "can", "walk", "explain")):
            q = var_prefix + q[0].lower() + q[1:]
        elif var_prefix and random.random() < 0.25:
            q = var_prefix + q

        # Ensure question ends with question mark
        q = q.strip()
        if not q.endswith("?"):
            q += "?"

        questions.add(q)

    return list(questions)[:target_count]


def generate_full_dataset(target_per_category: int = 720) -> pd.DataFrame:
    """
    Generate balanced dataset across all 5 classes.
    720 * 5 = 3,600 questions (well within 3,000-5,000 target).
    """
    print(f"[Dataset Generator] Generating {target_per_category} questions per category across {len(TOPIC_CATEGORIES)} categories...")
    all_rows = []
    
    for cat in TOPIC_CATEGORIES:
        cat_questions = generate_category_questions(cat, target_per_category)
        print(f"  - Category: '{cat}' generated {len(cat_questions)} unique questions.")
        for q in cat_questions:
            all_rows.append({"question": q, "label": cat})

    df = pd.DataFrame(all_rows)
    # Shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def split_dataset(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15):
    """
    Stratified split into 70% train, 15% validation, 15% test.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Splits must sum to 1.0"
    
    # First split: train vs temp (val + test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df["label"],
        random_state=42,
    )
    
    # Second split: val vs test (50/50 of temp)
    val_share = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_share),
        stratify=temp_df["label"],
        random_state=42,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def main():
    ensure_directories()
    
    # 1. Generate Raw Dataset
    raw_df = generate_full_dataset(target_per_category=720)
    raw_df.to_csv(RAW_QUESTIONS_PATH, index=False)
    print(f"\n[Raw Data] Saved {len(raw_df)} questions to {RAW_QUESTIONS_PATH}")

    # 2. Preprocessing & Deduplication
    processed_df = clean_dataset(raw_df.copy(), text_col="question", label_col="label")
    processed_df.to_csv(LABELED_QUESTIONS_PATH, index=False)
    print(f"[Processed Data] Saved {len(processed_df)} cleaned questions to {LABELED_QUESTIONS_PATH}")

    # 3. Stratified Train / Validation / Test Splits (70 / 15 / 15)
    train_df, val_df, test_df = split_dataset(processed_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    train_df.to_csv(TRAIN_SPLIT_PATH, index=False)
    val_df.to_csv(VAL_SPLIT_PATH, index=False)
    test_df.to_csv(TEST_SPLIT_PATH, index=False)

    print("\n[Splits Generated]")
    print(f"  - Train split (70%):      {len(train_df)} samples -> {TRAIN_SPLIT_PATH}")
    print(f"  - Validation split (15%): {len(val_df)} samples -> {VAL_SPLIT_PATH}")
    print(f"  - Test split (15%):       {len(test_df)} samples -> {TEST_SPLIT_PATH}")

    print("\n[Class Distribution in Train]")
    print(train_df["label"].value_counts())
    print("\n[Class Distribution in Test]")
    print(test_df["label"].value_counts())
    print("\n[Data Layer Complete] All datasets and splits generated successfully.")


if __name__ == "__main__":
    main()
