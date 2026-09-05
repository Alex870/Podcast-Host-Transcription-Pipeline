from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "Configure-PodcastTranscribeReviewBackend.ps1"
ROOT_LAUNCHER = PROJECT_ROOT / "Run Podcast Transcribe.ps1"
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")


@unittest.skipUnless(POWERSHELL, "PowerShell is required for launcher tests")
class ReviewBackendConfiguratorTests(unittest.TestCase):
    def run_powershell(self, body: str) -> str:
        script_literal = str(SCRIPT_PATH).replace("'", "''")
        command = (
            f". '{script_literal}' -LibraryOnly -NoPause\n"
            "$ErrorActionPreference = 'Stop'\n"
            f"{body}\n"
        )
        result = subprocess.run(
            [POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", command],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            self.fail(
                f"PowerShell failed with {result.returncode}:\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        return result.stdout.strip()

    def test_normalizes_supported_server_inputs(self) -> None:
        output = self.run_powershell(
            """
@(
    Normalize-ReviewServerAddress '192.168.1.230'
    Normalize-ReviewServerAddress '192.168.1.230:8000'
    Normalize-ReviewServerAddress 'my-vllm-server:8000'
    Normalize-ReviewServerAddress 'http://my-server:8000/v1'
    Normalize-ReviewServerAddress 'https://my-server'
) | ConvertTo-Json -Compress
"""
        )
        self.assertEqual(
            json.loads(output),
            [
                "http://192.168.1.230",
                "http://192.168.1.230:8000",
                "http://my-vllm-server:8000",
                "http://my-server:8000",
                "https://my-server",
            ],
        )

    def test_candidate_ports_prefer_current_then_standard_ports(self) -> None:
        output = self.run_powershell(
            """
@(Get-ReviewServerCandidates 'review-host' 'http://review-host:9100') |
    ConvertTo-Json -Compress
"""
        )
        self.assertEqual(
            json.loads(output),
            [
                "http://review-host:9100",
                "http://review-host:8000",
                "http://review-host:1234",
            ],
        )

    def test_detects_vllm_and_parses_model_ids(self) -> None:
        output = self.run_powershell(
            """
$request = {
    param($Method, $Uri, $Body)
    if ($Uri -like '*/api/v1/models') {
        throw 'not found'
    }
    if ($Uri -like '*/v1/models') {
        return [pscustomobject]@{
            data = @(
                [pscustomobject]@{ id = 'model-b' },
                [pscustomobject]@{ id = 'model-a' }
            )
        }
    }
    if ($Uri -like '*/version') {
        return [pscustomobject]@{ version = '0.10.0' }
    }
    throw 'not found'
}
$result = Find-ReviewServer -Candidates @('http://server:8000') -RequestInvoker $request
$result | ConvertTo-Json -Depth 10 -Compress
"""
        )
        result = json.loads(output)
        self.assertTrue(result["Succeeded"])
        self.assertEqual(result["Backend"], "vllm")
        self.assertEqual(result["ModelIds"], ["model-a", "model-b"])

    def test_detects_lm_studio_and_selected_model_completion(self) -> None:
        output = self.run_powershell(
            """
$request = {
    param($Method, $Uri, $Body)
    if ($Uri -like '*/api/v1/models') {
        return [pscustomobject]@{
            models = @(
                [pscustomobject]@{ key = 'local-model'; type = 'llm' },
                [pscustomobject]@{ key = 'embedding-model'; type = 'embedding' }
            )
        }
    }
    if ($Uri -like '*/v1/models') {
        return [pscustomobject]@{ data = @([pscustomobject]@{ id = 'local-model' }) }
    }
    if ($Uri -like '*/v1/chat/completions') {
        return [pscustomobject]@{
            choices = @([pscustomobject]@{
                message = [pscustomobject]@{ content = 'OK' }
            })
        }
    }
    throw 'not found'
}
$discovery = Find-ReviewServer -Candidates @('http://localhost:1234') -RequestInvoker $request
$completion = Test-ReviewChatCompletion -BaseUrl $discovery.BaseUrl `
    -ModelId 'local-model' -RequestInvoker $request
[pscustomobject]@{
    backend = $discovery.Backend
    completion = $completion.Succeeded
    models = @($discovery.ModelIds)
} | ConvertTo-Json -Compress
"""
        )
        result = json.loads(output)
        self.assertEqual(result["backend"], "lm_studio")
        self.assertTrue(result["completion"])
        self.assertEqual(result["models"], ["local-model"])

    def test_atomic_write_preserves_unrelated_keys_and_uses_utf8_without_bom(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "podcast_transcribe_config.json"
            original = {
                "hf_token": "secret-value",
                "default_source_dir": "source",
                "backend": "none",
                "review_base_url": "",
                "review_model_name": "",
                "transcript_cleanup_review": False,
            }
            config_path.write_text(json.dumps(original, indent=2), encoding="utf-8")
            path_literal = str(config_path).replace("'", "''")
            output = self.run_powershell(
                f"""
$changes = [ordered]@{{
    backend = 'vllm'
    review_base_url = 'http://server:8000'
    review_model_name = 'selected-model'
}}
$backup = Write-ReviewBackendConfiguration -ConfigPath '{path_literal}' -Changes $changes
[pscustomobject]@{{ backup = $backup }} | ConvertTo-Json -Compress
"""
            )
            result = json.loads(output)
            updated_bytes = config_path.read_bytes()
            updated = json.loads(updated_bytes.decode("utf-8"))
            backup_path = Path(result["backup"])

            self.assertFalse(updated_bytes.startswith(b"\xef\xbb\xbf"))
            self.assertEqual(updated["hf_token"], "secret-value")
            self.assertEqual(updated["default_source_dir"], "source")
            self.assertEqual(updated["backend"], "vllm")
            self.assertEqual(updated["review_model_name"], "selected-model")
            self.assertFalse(updated["transcript_cleanup_review"])
            self.assertTrue(backup_path.is_file())
            self.assertEqual(json.loads(backup_path.read_text("utf-8")), original)

    def test_review_modes_only_change_stages_when_explicitly_selected(self) -> None:
        output = self.run_powershell(
            """
$config = [pscustomobject]@{
    runtime_profile = 'custom'
    transcript_cleanup_review = $false
    glossary_correction_review = $false
    speaker_consistency_review = $false
    episode_qa_review = $false
}
$keep = Get-ConfigurationChanges -Config $config -Backend vllm `
    -BaseUrl 'http://server:8000' -ModelId model -ReviewMode Keep
$local = Get-ConfigurationChanges -Config $config -Backend vllm `
    -BaseUrl 'http://server:8000' -ModelId model -ReviewMode Local
$all = Get-ConfigurationChanges -Config $config -Backend vllm `
    -BaseUrl 'http://server:8000' -ModelId model -ReviewMode All
[pscustomobject]@{
    keepKeys = @($keep.Keys)
    local = $local
    all = $all
} | ConvertTo-Json -Depth 10 -Compress
"""
        )
        result = json.loads(output)
        self.assertEqual(
            result["keepKeys"],
            ["backend", "review_base_url", "review_model_name"],
        )
        self.assertNotIn("runtime_profile", result["local"])
        self.assertTrue(result["local"]["transcript_cleanup_review"])
        self.assertFalse(result["local"]["episode_qa_review"])
        self.assertEqual(result["all"]["runtime_profile"], "high_context_5090")
        self.assertTrue(result["all"]["episode_qa_review"])

    def test_cancellation_before_discovery_leaves_config_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "podcast_transcribe_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "backend": "none",
                        "review_base_url": "",
                        "review_model_name": "",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            before = hashlib.sha256(config_path.read_bytes()).hexdigest()
            root_literal = str(root).replace("'", "''")
            self.run_powershell(
                f"""
function global:Read-Host {{ param([string]$Prompt) return 'Q' }}
$null = Invoke-ReviewBackendWizard -RootPath '{root_literal}'
"""
            )
            after = hashlib.sha256(config_path.read_bytes()).hexdigest()
            self.assertEqual(after, before)
            self.assertEqual(
                list(root.glob("podcast_transcribe_config.backup-*.json")),
                [],
            )

    def test_final_confirmation_cancellation_leaves_config_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "podcast_transcribe_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "backend": "none",
                        "review_base_url": "",
                        "review_model_name": "",
                        "hf_token": "preserve-me",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            before = hashlib.sha256(config_path.read_bytes()).hexdigest()
            root_literal = str(root).replace("'", "''")
            self.run_powershell(
                f"""
$script:responses = New-Object 'System.Collections.Generic.Queue[string]'
@('http://server:8000', 'Y', '1', '1', 'N') |
    ForEach-Object {{ $script:responses.Enqueue($_) }}
function global:Read-Host {{ param([string]$Prompt) return $script:responses.Dequeue() }}
function global:Find-ReviewServer {{
    return [pscustomobject]@{{
        Succeeded = $true
        BaseUrl = 'http://server:8000'
        Backend = 'vllm'
        ModelIds = @('model-a')
        Attempts = @()
    }}
}}
function global:Test-ReviewChatCompletion {{
    return [pscustomobject]@{{ Succeeded = $true; Detail = 'ok' }}
}}
$null = Invoke-ReviewBackendWizard -RootPath '{root_literal}'
"""
            )
            after = hashlib.sha256(config_path.read_bytes()).hexdigest()
            self.assertEqual(after, before)
            self.assertEqual(
                list(root.glob("podcast_transcribe_config.backup-*.json")),
                [],
            )

    def test_failed_completion_test_leaves_config_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "podcast_transcribe_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "backend": "none",
                        "review_base_url": "",
                        "review_model_name": "",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            before = hashlib.sha256(config_path.read_bytes()).hexdigest()
            root_literal = str(root).replace("'", "''")
            output = self.run_powershell(
                f"""
$script:responses = New-Object 'System.Collections.Generic.Queue[string]'
@('http://server:8000', 'Y', '1') |
    ForEach-Object {{ $script:responses.Enqueue($_) }}
function global:Read-Host {{ param([string]$Prompt) return $script:responses.Dequeue() }}
function global:Find-ReviewServer {{
    return [pscustomobject]@{{
        Succeeded = $true
        BaseUrl = 'http://server:8000'
        Backend = 'vllm'
        ModelIds = @('model-a')
        Attempts = @()
    }}
}}
function global:Test-ReviewChatCompletion {{
    return [pscustomobject]@{{ Succeeded = $false; Detail = 'mock completion failure' }}
}}
try {{
    $null = Invoke-ReviewBackendWizard -RootPath '{root_literal}'
}} catch {{
    'EXPECTED_FAILURE'
}}
"""
            )
            after = hashlib.sha256(config_path.read_bytes()).hexdigest()
            self.assertIn("EXPECTED_FAILURE", output)
            self.assertEqual(after, before)
            self.assertEqual(
                list(root.glob("podcast_transcribe_config.backup-*.json")),
                [],
            )

    def test_root_launcher_exposes_option_seven_and_direct_action(self) -> None:
        launcher = ROOT_LAUNCHER.read_text(encoding="utf-8-sig")
        self.assertIn('"ConfigureLLM"', launcher)
        self.assertIn('Write-Host "  7. Configure external review LLM"', launcher)
        self.assertIn('"7" { return "ConfigureLLM" }', launcher)
        self.assertIn("while ($true)", launcher)
        self.assertIn('if ($selectedAction -eq "Quit")', launcher)
        self.assertIn('Write-Host "Returning to the main menu..."', launcher)

    def test_root_launcher_exposes_anonymous_meeting_profile(self) -> None:
        launcher = ROOT_LAUNCHER.read_text(encoding="utf-8-sig")
        self.assertIn('Write-Host "  9. Transcribe committee meeting (anonymous speakers)"', launcher)
        self.assertIn('"9" { return "AnonymousMeeting" }', launcher)
        self.assertIn('"AnonymousMeeting" { Invoke-LauncherScript -Path $RunScript -WorkflowProfile anonymous_meeting }', launcher)

    def test_interactive_launcher_returns_to_menu_after_action(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scripts = root / "scripts"
            scripts.mkdir()
            launcher_path = root / ROOT_LAUNCHER.name
            launcher_path.write_text(
                ROOT_LAUNCHER.read_text(encoding="utf-8-sig"),
                encoding="utf-8",
            )
            (scripts / "Debug-PodcastTranscribeEnvironment.ps1").write_text(
                'Write-Host "MOCK_ACTION_COMPLETED"\n',
                encoding="utf-8",
            )
            result = subprocess.run(
                [POWERSHELL, "-NoProfile", "-File", str(launcher_path)],
                input="1\nQ\n",
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("MOCK_ACTION_COMPLETED", result.stdout)
            self.assertIn("Returning to the main menu...", result.stdout)
            self.assertEqual(
                result.stdout.count("Podcast Host Transcription Pipeline"),
                2,
            )


if __name__ == "__main__":
    unittest.main()
