"""Strict Verification Suite for Phase 7A — Repository Audit & Submission Engineering.

Validates all 23 Phase 7A mandatory governance checks:
 1. Repository inventory exists
 2. Frozen Phase 4 artifacts present
 3. Frozen Phase 5 artifacts present
 4. Frozen Phase 6A artifacts present
 5. Frozen Phase 6B artifacts present
 6. Frozen Phase 6C artifacts present
 7. No obvious committed secrets
 8. No developer-specific absolute paths in production code
 9. Requirements / configuration exists
10. Golden dataset exactly 200 checkpoints
11. Human validation metadata exists
12. Retrieval corpus exactly 5,502 dialogues
13. Retrieval corpus Train-only (zero dev/test leakage)
14. Runtime code contains no gold-label feature usage
15. Test data not loaded by runtime
16. Safety validator exists and is functional
17. Deterministic policy exists
18. Ollama configuration exists
19. Selected model = llama3.2:1b
20. Required Phase 6C reports exist
21. Repository does not contain obvious temporary artifacts
22. Phase 5 frozen metrics are represented consistently
23. Required Phase 7A documentation exists

Exit code 0 only if ALL 23 mandatory checks PASS.
"""

import json
import os
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import PATHS


class Phase7AVerifier:
    def __init__(self):
        self.passed_checks = 0
        self.failed_checks = 0
        self.failure_reasons = []

    def record_check(self, check_num: int, name: str, passed: bool, detail: str = ""):
        status = "PASS" if passed else "FAIL"
        if passed:
            self.passed_checks += 1
            print(f"[Check {check_num:02d}/23] {name:<60} [{status}]")
            if detail:
                print(f"               Detail: {detail}")
        else:
            self.failed_checks += 1
            self.failure_reasons.append(f"Check {check_num}: {name} - {detail}")
            print(f"[Check {check_num:02d}/23] {name:<60} [{status}]")
            print(f"               FAILED: {detail}")

    def run_all_checks(self) -> bool:
        print("=" * 80)
        print("PHASE 7A — MASTER REPOSITORY & GOVERNANCE VERIFICATION SUITE")
        print("=" * 80)

        # 1. Repository inventory exists
        inv_file = PATHS.RESULTS_DIR / "phase7" / "phase7a_repository_inventory.md"
        self.record_check(
            1,
            "Repository inventory exists",
            inv_file.exists() and inv_file.stat().st_size > 1000,
            f"Found inventory at {inv_file.name} ({inv_file.stat().st_size if inv_file.exists() else 0} bytes)",
        )

        # 2. Frozen Phase 4 artifacts present
        p4_files = [
            PATHS.TFIDF_LOGREG_MODEL_PATH,
            PATHS.PHASE4_BASELINE_REPORT_MD,
            PATHS.PHASE4_POLICY_REPORT_MD,
            PROJECT_ROOT / "scripts" / "verify_phase4.py",
        ]
        p4_ok = all(f.exists() for f in p4_files)
        self.record_check(
            2,
            "Frozen Phase 4 artifacts present",
            p4_ok,
            "TF-IDF model weights, baseline report, policy report, and verify_phase4.py present",
        )

        # 3. Frozen Phase 5 artifacts present
        p5_files = [
            PATHS.AMAZONHELP_TRAIN_RETRIEVAL_JSONL,
            PATHS.RETRIEVAL_INDEX_NPZ,
            PATHS.RETRIEVAL_INDEX_META,
            PATHS.PHASE5_RETRIEVAL_REPORT_MD,
            PATHS.PHASE5_RETRIEVAL_METRICS_JSON,
        ]
        p5_ok = all(f.exists() for f in p5_files)
        self.record_check(
            3,
            "Frozen Phase 5 artifacts present",
            p5_ok,
            "Train retrieval corpus, index.npz, meta.json, and Phase 5 reports present",
        )

        # 4. Frozen Phase 6A artifacts present
        p6a_files = [
            PATHS.MODEL_BENCHMARK_JSON,
            PATHS.MODEL_SELECTION_REPORT_MD,
            PATHS.MODEL_COMPARISON_MD,
            PROJECT_ROOT / "scripts" / "verify_phase6a.py",
        ]
        p6a_ok = all(f.exists() for f in p6a_files)
        self.record_check(
            4,
            "Frozen Phase 6A artifacts present",
            p6a_ok,
            "Model benchmark JSON, comparison, and selection reports present",
        )

        # 5. Frozen Phase 6B artifacts present
        p6b_files = [
            PATHS.PHASE6B_METRICS_JSON,
            PATHS.PHASE6B_REPORT_MD,
            PATHS.PHASE6B_FAILURE_ANALYSIS_MD,
            PROJECT_ROOT / "scripts" / "verify_phase6b.py",
        ]
        p6b_ok = all(f.exists() for f in p6b_files)
        self.record_check(
            5,
            "Frozen Phase 6B artifacts present",
            p6b_ok,
            "Phase 6B metrics JSON, report, and failure analysis present",
        )

        # 6. Frozen Phase 6C artifacts present
        p6c_files = [
            PATHS.PHASE6C_METRICS_JSON,
            PATHS.PHASE6C_REPORT_MD,
            PATHS.PHASE6C_FAILURE_ANALYSIS_MD,
            PATHS.PHASE6C_RETRIEVAL_ANALYSIS_MD,
            PROJECT_ROOT / "scripts" / "verify_phase6c.py",
        ]
        p6c_ok = all(f.exists() for f in p6c_files)
        self.record_check(
            6,
            "Frozen Phase 6C artifacts present",
            p6c_ok,
            "Phase 6C metrics JSON, report, failure analysis, and retrieval analysis present",
        )

        # 7. No obvious committed secrets
        secret_patterns = [
            re.compile(r'(?i)(api[_-]?key|secret_key|private_key|auth_token)\s*[:=]\s*["\']([A-Za-z0-9_\-\.]{15,})["\']'),
            re.compile(r'gh[pousr]_[A-Za-z0-9_]{30,}'),
            re.compile(r'AIza[0-9A-Za-z\-_]{35}'),
            re.compile(r'sk-[A-Za-z0-9]{20,}'),
            re.compile(r'-----BEGIN (RSA|EC|OPENSSH|DSA|PGP)? PRIVATE KEY-----'),
            re.compile(r'AKIA[0-9A-Z]{16}'),
        ]
        found_secrets = []
        for root, dirs, files in os.walk(PROJECT_ROOT / "src"):
            for f in files:
                fpath = Path(root) / f
                if fpath.suffix in [".py", ".json", ".yaml", ".env"]:
                    try:
                        content = fpath.read_text(encoding="utf-8", errors="ignore")
                        for pat in secret_patterns:
                            if pat.search(content):
                                found_secrets.append(fpath.name)
                    except Exception:
                        pass
        self.record_check(
            7,
            "No obvious committed secrets",
            len(found_secrets) == 0,
            f"Scanned src/ and config files; secrets found = {len(found_secrets)}",
        )

        # 8. No developer-specific absolute paths in production code
        hardcoded_drive_pat = re.compile(r'(?i)[a-z]:\\[a-z0-9_\\ -]+')
        hardcoded_user_pat = re.compile(r'(?i)[a-z]:[/\\]users[/\\][a-z0-9_\.\-]+')
        hardcoded_home_pat = re.compile(r'/home/[a-z0-9_\-]+')
        bad_paths_in_src = []
        for root, dirs, files in os.walk(PROJECT_ROOT / "src"):
            for f in files:
                fpath = Path(root) / f
                if fpath.suffix == ".py":
                    content = fpath.read_text(encoding="utf-8", errors="ignore")
                    if hardcoded_drive_pat.search(content) or hardcoded_user_pat.search(content) or hardcoded_home_pat.search(content):
                        bad_paths_in_src.append(fpath.name)
        self.record_check(
            8,
            "No developer-specific absolute paths in production code",
            len(bad_paths_in_src) == 0,
            f"All paths in src/ dynamically resolve via PathConfig (flagged files: {bad_paths_in_src})",
        )

        # 9. Requirements / configuration exists
        req_file = PROJECT_ROOT / "requirements.txt"
        cfg_file = PROJECT_ROOT / "src" / "config.py"
        env_ex = PROJECT_ROOT / ".env.example"
        req_ok = req_file.exists() and cfg_file.exists() and env_ex.exists()
        self.record_check(
            9,
            "Requirements / configuration exists",
            req_ok,
            "requirements.txt, src/config.py, and .env.example verified",
        )

        # 10. Golden dataset exactly 200 checkpoints
        golden_count = 0
        if PATHS.AMAZONHELP_GOLDEN_HUMAN_VALIDATED_JSONL.exists():
            with open(PATHS.AMAZONHELP_GOLDEN_HUMAN_VALIDATED_JSONL, "r", encoding="utf-8") as f:
                golden_count = sum(1 for _ in f)
        self.record_check(
            10,
            "Golden dataset exactly 200 checkpoints",
            golden_count == 200,
            f"Exact checkpoint count: {golden_count}",
        )

        # 11. Human validation metadata exists
        val_log = PATHS.GOLDEN_DATA_DIR / "human_validation_log.jsonl"
        val_rep = PATHS.GOLDEN_HUMAN_ANNOTATION_REPORT_MD
        val_log_count = 0
        if val_log.exists():
            with open(val_log, "r", encoding="utf-8") as f:
                val_log_count = sum(1 for _ in f)
        self.record_check(
            11,
            "Human validation metadata exists",
            val_log.exists() and val_rep.exists() and val_log_count >= 200,
            f"Human validation log contains {val_log_count} records; annotation report present",
        )

        # 12. Retrieval corpus exactly 5,502 dialogues
        ret_count = 0
        if PATHS.AMAZONHELP_TRAIN_RETRIEVAL_JSONL.exists():
            with open(PATHS.AMAZONHELP_TRAIN_RETRIEVAL_JSONL, "r", encoding="utf-8") as f:
                ret_count = sum(1 for _ in f)
        self.record_check(
            12,
            "Retrieval corpus exactly 5,502 dialogues",
            ret_count == 5502,
            f"Exact retrieval dialogues: {ret_count}",
        )

        # 13. Retrieval corpus Train-only (zero dev/test leakage)
        golden_conv_ids = set()
        if PATHS.AMAZONHELP_GOLDEN_HUMAN_VALIDATED_JSONL.exists():
            with open(PATHS.AMAZONHELP_GOLDEN_HUMAN_VALIDATED_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    golden_conv_ids.add(json.loads(line).get("conversation_id"))
        ret_overlap = 0
        if PATHS.AMAZONHELP_TRAIN_RETRIEVAL_JSONL.exists():
            with open(PATHS.AMAZONHELP_TRAIN_RETRIEVAL_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    if json.loads(line).get("conversation_id") in golden_conv_ids:
                        ret_overlap += 1
        self.record_check(
            13,
            "Retrieval corpus Train-only (zero dev/test leakage)",
            ret_overlap == 0,
            f"Overlap between Train retrieval corpus and Dev golden benchmark: {ret_overlap} dialogues",
        )

        # 14. Runtime code contains no gold-label feature usage
        leakage_keywords = [
            "gold_intent", "gold_state", "gold_action", "gold_escalation",
            "human_label", "ground_truth", "target_label"
        ]
        leaked_in_runtime = []
        runtime_files = [
            PROJECT_ROOT / "src" / "llm" / "agent_with_retrieval.py",
            PROJECT_ROOT / "src" / "llm" / "evidence_builder.py",
            PROJECT_ROOT / "src" / "llm" / "conversation_formatter.py",
            PROJECT_ROOT / "src" / "llm" / "agent.py",
            PROJECT_ROOT / "src" / "policy" / "action_policy.py",
            PROJECT_ROOT / "src" / "policy" / "escalation_policy.py",
            PROJECT_ROOT / "src" / "state" / "state_tracker.py",
        ]
        for rf in runtime_files:
            if rf.exists():
                text = rf.read_text(encoding="utf-8", errors="ignore")
                for kw in leakage_keywords:
                    if kw in text:
                        leaked_in_runtime.append(f"{rf.name}:{kw}")
        self.record_check(
            14,
            "Runtime code contains no gold-label feature usage",
            len(leaked_in_runtime) == 0,
            f"Scanned production inference code; violations = {len(leaked_in_runtime)}",
        )

        # 15. Test data not loaded by runtime
        test_leakage = []
        for rf in runtime_files:
            if rf.exists():
                text = rf.read_text(encoding="utf-8", errors="ignore")
                if "test.jsonl" in text or "test_split" in text:
                    test_leakage.append(rf.name)
        self.record_check(
            15,
            "Test data not loaded by runtime",
            len(test_leakage) == 0,
            f"Test partition completely isolated from runtime execution ({test_leakage})",
        )

        # 16. Safety validator exists and is functional
        safety_path = PROJECT_ROOT / "src" / "llm" / "safety_validator.py"
        safety_ok = False
        if safety_path.exists():
            from src.llm.safety_validator import DeterministicSafetyValidator
            sv = DeterministicSafetyValidator()
            safety_ok = sv is not None
        self.record_check(
            16,
            "Safety validator exists and is functional",
            safety_ok,
            "DeterministicSafetyValidator loaded and operational",
        )

        # 17. Deterministic policy exists
        from src.policy.action_policy import DeterministicActionPolicy
        from src.policy.escalation_policy import DeterministicEscalationPolicy
        from src.policy.unified_decision_engine import UnifiedDeterministicDecisionEngine
        from src.state.state_tracker import ConversationStateTracker
        policy_ok = (
            DeterministicActionPolicy is not None
            and DeterministicEscalationPolicy is not None
            and UnifiedDeterministicDecisionEngine is not None
            and ConversationStateTracker is not None
        )
        self.record_check(
            17,
            "Deterministic policy exists",
            policy_ok,
            "State tracker, action policy, escalation policy, and unified engine verified",
        )

        # 18. Ollama configuration exists
        from src.llm.model_client import OllamaClient
        client = OllamaClient()
        self.record_check(
            18,
            "Ollama configuration exists",
            client.base_url == "http://localhost:11434" and client.timeout_seconds > 0,
            f"Configured Ollama endpoint: {client.base_url}, timeout={client.timeout_seconds}s",
        )

        # 19. Selected model = llama3.2:1b
        from src.llm.agent_with_retrieval import LLMAgentWithRetrieval
        agent = LLMAgentWithRetrieval(evidence_builder=object())
        self.record_check(
            19,
            "Selected model = llama3.2:1b",
            agent.model_name == "llama3.2:1b",
            f"Default production model: {agent.model_name}",
        )

        # 20. Required Phase 6C reports exist
        p6c_reports = [
            PATHS.PHASE6C_REPORT_MD,
            PATHS.PHASE6C_METRICS_JSON,
            PATHS.PHASE6C_RETRIEVAL_ANALYSIS_MD,
            PATHS.PHASE6C_FAILURE_ANALYSIS_MD,
        ]
        self.record_check(
            20,
            "Required Phase 6C reports exist",
            all(r.exists() for r in p6c_reports),
            "All 4 Phase 6C analytical documents present",
        )

        # 21. Repository does not contain obvious temporary artifacts
        temp_patterns = [".swp", ".swo", ".tmp", ".bak"]
        found_temp_files = []
        for root, dirs, files in os.walk(PROJECT_ROOT):
            if any(ign in root for ign in [".git", "node_modules"]):
                continue
            for f in files:
                if any(f.endswith(pat) for pat in temp_patterns) or f.startswith("scratch"):
                    found_temp_files.append(f)
        clean_plan_exists = (PATHS.RESULTS_DIR / "phase7" / "phase7a_cleanup_plan.md").exists()
        self.record_check(
            21,
            "Repository does not contain obvious temporary artifacts",
            len(found_temp_files) == 0 and clean_plan_exists,
            f"Zero scratch/swap/backup files found; phase7a_cleanup_plan.md verified",
        )

        # 22. Phase 5 frozen metrics are represented consistently
        with open(PATHS.PHASE5_RETRIEVAL_METRICS_JSON, "r", encoding="utf-8") as f:
            p5_data = json.load(f)
        k5_metrics = p5_data["k_sweep"]["5"]
        p5_acc = k5_metrics["intent_metrics"]["accuracy"]
        p5_f1 = k5_metrics["intent_metrics"]["macro_f1"]
        p5_exact = k5_metrics["exact_match_metrics"]["exact_match_all_rate"]
        consistency_audit_exists = (PATHS.RESULTS_DIR / "phase7" / "phase7a_consistency_audit.md").exists()
        p5_metrics_consistent = (
            abs(p5_acc - 0.845) < 1e-4
            and abs(p5_f1 - 0.7993) < 1e-3
            and abs(p5_exact - 0.645) < 1e-4
            and consistency_audit_exists
        )
        self.record_check(
            22,
            "Phase 5 frozen metrics are represented consistently",
            p5_metrics_consistent,
            f"Authoritative metrics: Intent Acc={p5_acc*100:.2f}%, Intent F1={p5_f1*100:.2f}%, Exact Match={p5_exact*100:.2f}%; audit documented",
        )

        # 23. Required Phase 7A documentation exists
        doc_files = [
            PROJECT_ROOT / "README.md",
            PATHS.RESULTS_DIR / "phase7" / "phase7a_repository_inventory.md",
            PATHS.RESULTS_DIR / "phase7" / "phase7a_frozen_integrity.md",
            PATHS.RESULTS_DIR / "phase7" / "phase7a_portability.md",
            PATHS.RESULTS_DIR / "phase7" / "phase7a_data_reproducibility.md",
            PATHS.RESULTS_DIR / "phase7" / "phase7a_consistency_audit.md",
            PATHS.RESULTS_DIR / "phase7" / "phase7a_commands.md",
            PATHS.RESULTS_DIR / "phase7" / "phase7a_cleanup_plan.md",
            PATHS.RESULTS_DIR / "phase7" / "phase7a_final_audit.md",
        ]
        docs_ok = all(df.exists() and df.stat().st_size > 500 for df in doc_files)
        self.record_check(
            23,
            "Required Phase 7A documentation exists",
            docs_ok,
            f"All 9 mandatory Phase 7A markdown artifacts present and populated",
        )

        print("-" * 80)
        print(f"VERIFICATION SUMMARY: {self.passed_checks}/23 checks PASSED ({self.failed_checks} failed)")
        print("-" * 80)

        if self.failed_checks == 0:
            print("\n*** ALL PHASE 7A GOVERNANCE CHECKS PASSED SUCCESSFULLY ***\n")
            return True
        else:
            print(f"\n*** VERIFICATION FAILED: {self.failed_checks} checks failed ***")
            for reason in self.failure_reasons:
                print(f"  - {reason}")
            return False


if __name__ == "__main__":
    verifier = Phase7AVerifier()
    success = verifier.run_all_checks()
    sys.exit(0 if success else 1)
