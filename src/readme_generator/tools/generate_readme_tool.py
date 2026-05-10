from crewai.tools import tool
import json
import re
from typing import Any, Dict, List
from .chatopenai import LLM_Callable
from .common_utils import infer_family_hint_from_corpus, is_url_source_mode, normalize_list


class GenerateReadmeTool:
    global_memory = None
    # Transient pipeline state passed between step_* tools (not persisted to GlobalMemory).
    _pipeline: Dict[str, Any] = {}
    llm = LLM_Callable(
        base_url="http://10.54.34.78:30000/v1",
        api_key="empty",
        model_name="local-model",
    )

    @staticmethod
    def _normalize_js_files(js_files: list) -> list:
        normalized = []
        for i, item in enumerate(js_files or []):
            if isinstance(item, dict):
                path = str(item.get("path") or f"file_{i}.js")
                content = str(item.get("content") or "")
            else:
                path = f"file_{i}.js"
                content = str(item or "")
            if not path.endswith(".js"):
                path = f"{path}.js"
            normalized.append({"path": path, "content": content})
        return normalized

    @staticmethod
    def _compose_family_content(family_md: str, family_index_js: str, family_js_files: list | None = None) -> str:
        md_text = (family_md or "").strip()
        js_text = (family_index_js or "").strip()
        js_files = GenerateReadmeTool._normalize_js_files(family_js_files or [])
        if js_files:
            sections = []
            for item in js_files:
                sections.append(
                    f"### {item['path']}\n\n```javascript\n{(item.get('content') or '').strip()}\n```"
                )
            js_block = "\n\n".join(sections).strip()
        else:
            js_block = f"### index.js\n\n```javascript\n{js_text}\n```" if js_text else ""

        if not md_text and not js_block:
            return ""
        if not js_block:
            return md_text
        if not md_text:
            return js_block
        return f"{md_text}\n\n---\n\n{js_block}"

    @staticmethod
    def _validate_target_models(family_md: str, family_index_js: str) -> None:
        mode = str(GenerateReadmeTool.global_memory.memory_retrieve("generation_mode") or "").strip().lower()
        model_list = GenerateReadmeTool.global_memory.memory_retrieve("model_list") or []
        model_id_list = GenerateReadmeTool.global_memory.memory_retrieve("model_id_list") or []
        all_text = f"{family_md or ''}\n{family_index_js or ''}"
        lowered = all_text.lower()

        candidates = []
        for raw in list(model_list) + list(model_id_list):
            text = str(raw or "").strip()
            if not text:
                continue
            candidates.append(text)
            if "/" in text:
                candidates.append(text.split("/", 1)[1].strip())

        deduped = []
        seen = set()
        for c in candidates:
            lc = c.lower()
            if lc in seen:
                continue
            seen.add(lc)
            deduped.append(c)

        if not deduped:
            return
        if not any(c.lower() in lowered for c in deduped):
            if is_url_source_mode(mode):
                family_hint = infer_family_hint_from_corpus(deduped) or infer_family_hint_from_corpus([all_text])
                if family_hint and family_hint in lowered:
                    return
            raise ValueError("Generated artifacts do not align with input model_list/model_id_list.")

    @staticmethod
    def _fix_md_component_references(md: str) -> str:
        """Replace <!-- 组件引用：ComponentName --> comments with proper MDX import + <Component /> usage.

        Docusaurus MDX requires import statements at the top of the file (before the first
        heading / content) and the component tag at the point of usage.
        """
        comment_pat = re.compile(r"<!--\s*组件引用[：:]\s*(\w+)\s*-->")
        refs = list(dict.fromkeys(comment_pat.findall(md)))  # unique, preserve order
        if not refs:
            return md

        # Replace each comment placeholder with the JSX self-closing tag.
        md = comment_pat.sub(lambda m: f"<{m.group(1)} />", md)

        # Collect import statements that are missing.
        missing_imports: List[str] = []
        for comp in refs:
            import_stmt = f"import {comp} from '@site/src/components/autoregressive/{comp}';"
            # Also allow double-quote variant already present.
            if import_stmt not in md and f'import {comp} from "@site/src/components/autoregressive/{comp}"' not in md:
                missing_imports.append(import_stmt)

        if missing_imports:
            imports_block = "\n".join(missing_imports) + "\n\n"
            # Place after YAML front matter (---) block if present, otherwise at the very top.
            if md.lstrip().startswith("---"):
                fm_start = md.index("---")
                fm_end = md.find("---", fm_start + 3)
                if fm_end != -1:
                    insert_at = fm_end + 3
                    while insert_at < len(md) and md[insert_at] in "\n\r":
                        insert_at += 1
                    md = md[:insert_at] + "\n\n" + imports_block + md[insert_at:]
                else:
                    md = imports_block + md
            else:
                md = imports_block + md

        return md

    @staticmethod
    def _fix_unclosed_tags(text: str) -> str:
        """Fix HTML/JSX closing tags that are missing their closing `>`.

        The LLM occasionally emits `</div` or `</span` (etc.) without the trailing `>`,
        which breaks JSX rendering. The `\\b` word boundary anchors the match to the full
        tag name, preventing backtracking into valid tags like `</small>`.
        """
        return re.sub(r'</([A-Za-z][A-Za-z0-9]*)\b(?!>)', r'</\1>', text)

    @staticmethod
    def _strip_think_blocks(text: str) -> str:
        s = str(text or "")
        start_tag = "<think>"
        end_tag = "</think>"
        while True:
            start = s.find(start_tag)
            if start < 0:
                break
            end = s.find(end_tag, start + len(start_tag))
            if end < 0:
                s = s[:start]
                break
            s = s[:start] + s[end + len(end_tag):]
        return s.strip()

    @staticmethod
    def _dedup_str_list(values: List[Any]) -> List[str]:
        out: List[str] = []
        seen = set()
        for item in normalize_list(values, fallback_single_str=True, stringify_items=True):
            s = str(item or "").strip()
            if not s:
                continue
            k = s.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(s)
        return out

    @staticmethod
    def _shrink_reference_text(text: str, head_chars: int = 1600, tail_chars: int = 400) -> str:
        s = str(text or "")
        if len(s) <= head_chars + tail_chars + 80:
            return s
        omitted = len(s) - head_chars - tail_chars
        return f"{s[:head_chars]}\n\n<!-- REF_TRUNCATED: omitted {omitted} chars -->\n\n{s[-tail_chars:]}"

    @staticmethod
    def _fallback_generate_from_reference(ctx: Dict[str, Any]) -> Dict[str, Any]:
        model_list = GenerateReadmeTool._dedup_str_list(ctx.get("model_list") or [])
        family_md = str(ctx.get("ref_md") or "").strip()
        family_index_js = str(ctx.get("ref_index_js") or "").strip()
        mode = str(ctx.get("generation_mode") or "").strip().lower()

        family_md, family_index_js = GenerateReadmeTool._align_reference_family_version(
            family_md=family_md,
            family_index_js=family_index_js,
            model_list=model_list,
            model_id_list=GenerateReadmeTool._dedup_str_list(ctx.get("model_id_list") or []),
        )

        if not family_md:
            family_md = "# Model Deployment\n\n"
        if model_list and not any(m.lower() in family_md.lower() for m in model_list):
            lines = [f"- {m}" for m in model_list]
            family_md = f"## Target Models\n\n{chr(10).join(lines)}\n\n{family_md}".strip()

        if is_url_source_mode(mode) and not ("intel" in family_md.lower() and "xeon" in family_md.lower()):
            # Try LLM-based unified rewrite first; fall back to minimal hardcoded sections.
            example_model = next(
                (m for m in model_list if "instruct" in m.lower() and "8b" in m.lower()
                 and "fp8" not in m.lower() and "w8a8" not in m.lower()),
                model_list[0] if model_list else "meta-llama/Llama-3.1-8B-Instruct",
            )
            llm_unified = None
            try:
                unified_prompt = f"""You are a technical documentation writer for LLM serving infrastructure.

Rewrite the README and index.js below so that CUDA, AMD, and Intel CPU (Xeon) are all
first-class peers at the same structural level inside ONE unified document each.
Do NOT simply append Intel content — restructure each section to cover all three backends
in parallel.

### Source family_md (CUDA/AMD only — restructure):
{GenerateReadmeTool._shrink_reference_text(family_md, head_chars=3000, tail_chars=800)}

### Source family_index_js:
{GenerateReadmeTool._shrink_reference_text(family_index_js, head_chars=2000, tail_chars=500)}

### Target models:
{json.dumps(model_list, ensure_ascii=False)}
### Example model: {example_model}

Intel Xeon: use `--device cpu`; single-socket `--tp 1`; dual-socket `--tp 2`.
Add `intelCpuBackend` object in JS mirroring existing backend style.

Output ONLY valid JSON (no fences):
{{"family_md": "<unified README>", "family_index_js": "<unified index.js>"}}"""
                response = GenerateReadmeTool.llm.invoke(unified_prompt)
                cleaned = GenerateReadmeTool._strip_think_blocks(response)
                cleaned_lower = cleaned.lstrip().lower()
                if not (cleaned_lower.startswith("<!doctype") or cleaned_lower.startswith("<html")):
                    parsed = json.loads(cleaned)
                    nm = str(parsed.get("family_md") or "").strip()
                    nj = str(parsed.get("family_index_js") or "").strip()
                    if nm and nj:
                        llm_unified = (nm, nj)
            except Exception as e:
                print(f"[fallback][llm_unify_failed] {e}")

            if llm_unified:
                family_md, family_index_js = llm_unified
            else:
                # Minimal hardcoded Intel section as last resort.
                intel_section = f"""## Intel CPU (Xeon) Deployment

SGLang supports running these models on Intel Xeon CPUs via the `--device cpu` flag.

### Launch Server (Single Socket)

```bash
python -m sglang.launch_server \\
  --model-path {example_model} \\
  --host 0.0.0.0 \\
  --tp 1 \\
  --device cpu
```

### Launch Server (Dual-Socket / Higher Throughput)

```bash
python -m sglang.launch_server \\
  --model-path {example_model} \\
  --host 0.0.0.0 \\
  --tp 2 \\
  --device cpu
```

### Benchmark (Intel Xeon CPU)

```bash
python -m sglang.bench_serving \\
  --dataset-name random \\
  --random-input-len 1024 \\
  --random-output-len 1024 \\
  --num-prompts 1 \\
  --max-concurrency 1 \\
  --request-rate inf
```""".strip()
                family_md = f"{family_md.rstrip()}\n\n{intel_section}\n"
                if "intel" not in family_index_js.lower():
                    family_index_js += (
                        "\n\n// Intel CPU (Xeon) backend\n"
                        "export const intelCpuBackend = {\n"
                        "  device: 'cpu',\n"
                        "  tpSizes: [1, 2, 4],\n"
                        f"  exampleModel: '{example_model}',\n"
                        "};\n"
                    )
        elif not family_index_js.strip():
            model_js = ", ".join([f'"{m}"' for m in model_list]) if model_list else ""
            family_index_js = (
                "export const modelList = ["
                + model_js
                + "];\n"
                + "export const backends = ['cuda', 'amd', 'intel_cpu'];\n"
                + "export const intelCpuBackend = { device: 'cpu', tpSizes: [1, 2, 4] };\n"
            )

        return {
            "family_md": family_md.strip(),
            "family_index_js": family_index_js.strip(),
            "family_js_files": [{"path": "index.js", "content": family_index_js.strip()}],
            "source": "fallback",
        }

    @staticmethod
    def _infer_target_family_version(model_list: List[str], model_id_list: List[str]) -> tuple[str, str]:
        texts = list(model_list or []) + list(model_id_list or [])
        for raw in texts:
            t = str(raw or "")
            m = re.search(r"\b(llama|qwen|gemma|mistral|deepseek|phi)\s*[-_ ]?(\d+(?:\.\d+)?)", t, flags=re.IGNORECASE)
            if m:
                return (m.group(1).lower(), m.group(2))
        return ("", "")

    @staticmethod
    def _infer_target_size(model_list: List[str], model_id_list: List[str]) -> str:
        texts = list(model_list or []) + list(model_id_list or [])
        for raw in texts:
            t = str(raw or "")
            m = re.search(r"[-_ ](\d+(?:\.\d+)?[bBmM])(?:[-_ ]|$)", t)
            if m:
                return m.group(1).upper()
        return ""

    @staticmethod
    def _align_reference_family_version(
        family_md: str,
        family_index_js: str,
        model_list: List[str],
        model_id_list: List[str],
    ) -> tuple[str, str]:
        fam, ver = GenerateReadmeTool._infer_target_family_version(model_list, model_id_list)
        if not fam or not ver:
            return (family_md, family_index_js)

        md = str(family_md or "")
        js = str(family_index_js or "")
        fam_title = fam.capitalize()

        if fam == "llama":
            md = re.sub(r"\bLlama\s+\d+(?:\.\d+)?\b", f"{fam_title} {ver}", md)
            md = re.sub(r"\bLlama-\d+(?:\.\d+)?\b", f"{fam_title}-{ver}", md)
            js = re.sub(r"\bLlama\s*\d+(?:\.\d+)?\b", f"{fam_title} {ver}", js)
            js = re.sub(r"\bLlama\d+(?:_\d+)?\b", f"{fam_title}{ver.replace('.', '')}", js)
        else:
            md = re.sub(
                rf"\b{fam_title}\s+\d+(?:\.\d+)?\b",
                f"{fam_title} {ver}",
                md,
                flags=re.IGNORECASE,
            )
            md = re.sub(
                rf"\b{fam_title}-\d+(?:\.\d+)?\b",
                f"{fam_title}-{ver}",
                md,
                flags=re.IGNORECASE,
            )
        return (md, js)

    @staticmethod
    def _normalize_artifacts_to_target_models(
        family_md: str,
        family_index_js: str,
        model_list: List[str],
        model_id_list: List[str],
    ) -> tuple[str, str]:
        md, js = GenerateReadmeTool._align_reference_family_version(
            family_md=family_md,
            family_index_js=family_index_js,
            model_list=model_list,
            model_id_list=model_id_list,
        )
        fam, ver = GenerateReadmeTool._infer_target_family_version(model_list, model_id_list)
        size = GenerateReadmeTool._infer_target_size(model_list, model_id_list)
        if fam == "llama" and ver:
            ver_pat = re.escape(ver)
            # fix lingering 3.1 references in links/paths/component names
            md = re.sub(r"meta-llama-3-1", f"meta-llama-{ver.replace('.', '-')}", md)
            md = re.sub(r"llama3_1", f"llama3_{ver.replace('.', '_')}", md, flags=re.IGNORECASE)
            js = re.sub(r"Llama31", f"Llama{ver.replace('.', '')}", js)
            # keep size aligned (e.g. 8B -> 3B) when target size is explicit
            if size:
                md = re.sub(rf"(Llama-{ver_pat}-)\d+(?:\.\d+)?[bBmM]", rf"\g<1>{size}", md)
                js = re.sub(rf"(Llama-{ver_pat}-)\d+(?:\.\d+)?[bBmM]", rf"\g<1>{size}", js)
                js = re.sub(rf"(Meta-Llama-{ver_pat}-)\d+(?:\.\d+)?[bBmM]", rf"\g<1>{size}", js)
        return (md, js)

    @staticmethod
    def _ensure_readme_command_content(
        family_md: str,
        family_index_js: str,
        model_list: List[str],
        model_id_list: List[str],
    ) -> str:
        md = str(family_md or "")
        js = str(family_index_js or "")
        # Drop truncation markers if they leak into generated family artifacts.
        md = re.sub(r"\n?\s*<!--\s*REF_TRUNCATED:[\s\S]*?-->\s*\n?", "\n\n", md, flags=re.IGNORECASE)
        lowered = md.lower()
        has_cmd_block = bool(re.search(r"```(?:bash|shell|sh|console)\s*[\s\S]*?```", md, flags=re.IGNORECASE))
        has_launch_cmd = "python -m sglang.launch_server" in lowered
        has_bench_cmd = "python -m sglang.bench_serving" in lowered
        if has_cmd_block and (has_launch_cmd or has_bench_cmd):
            return md.strip()

        target_model = (
            GenerateReadmeTool._dedup_str_list(model_id_list or [])
            or GenerateReadmeTool._dedup_str_list(model_list or [])
            or ["<MODEL_ID>"]
        )[0]
        tp_match = re.search(r"--tp\s+(\d+)", js)
        tp = tp_match.group(1) if tp_match else "1"
        cmd_section = f"""
## Quick Start Commands

### Launch Serving

```bash
python -m sglang.launch_server \\
  --model-path {target_model} \\
  --host 0.0.0.0 \\
  --tp {tp}
```

### Benchmark

```bash
python -m sglang.bench_serving \\
  --dataset-name random \\
  --random-input-len 1024 \\
  --random-output-len 1024 \\
  --num-prompts 1 \\
  --max-concurrency 1 \\
  --request-rate inf
```
""".strip()
        return f"{md.rstrip()}\n\n{cmd_section}\n".strip()

    @staticmethod
    def _llm_judge_generation_mode(ctx: Dict[str, Any]) -> str:
        compact_ctx = {
            "generation_mode": str(ctx.get("generation_mode") or ""),
            "model_list": ctx.get("model_list") or [],
            "github_url": ctx.get("github_url") or [],
            "source_md_url": ctx.get("source_md_url") or "",
            "source_js_url": ctx.get("source_js_url") or "",
            "github_md_folder_url": ctx.get("github_md_folder_url") or "",
            "github_js_folder_url": ctx.get("github_js_folder_url") or "",
            "source_md_files_count": len(ctx.get("source_md_files") or []),
            "source_js_files_count": len(ctx.get("source_js_files") or []),
            "remote_payload_generation_mode": str((ctx.get("remote_payload") or {}).get("generation_mode") or ""),
            "remote_payload_source_urls": (ctx.get("remote_payload") or {}).get("source_urls") or {},
        }
        prompt = f"""
You are a strict workflow mode classifier.
Classify the generation flow as:
- "reference" (legacy md/js adaptation)
- "web_sources" (url_source/github_folders flow)

Rules:
1. If source URLs or source file collections are the primary inputs, choose "web_sources".
2. If model_list + github_url/ref_md/ref_index_js are the primary inputs, choose "reference".
3. Prefer consistency with explicit generation_mode when evidence is not conflicting.
4. Output ONLY JSON: {{"generation_mode":"reference|web_sources"}}.

Input:
{json.dumps(compact_ctx, ensure_ascii=False)}
"""
        try:
            response = GenerateReadmeTool.llm.invoke(prompt)
            cleaned = GenerateReadmeTool._strip_think_blocks(response)
            parsed = json.loads(cleaned)
            mode = str(parsed.get("generation_mode") or "").strip().lower()
            if mode in {"reference", "web_sources"}:
                return mode
        except Exception:
            pass
        return ""

    @staticmethod
    def _resolve_generation_mode(ctx: Dict[str, Any]) -> tuple[str, str]:
        explicit_mode = str(ctx.get("generation_mode") or "").strip().lower()
        remote_payload = ctx.get("remote_payload") or {}
        remote_mode = str(remote_payload.get("generation_mode") or "").strip().lower()

        model_list = GenerateReadmeTool._dedup_str_list(ctx.get("model_list") or [])
        github_url = GenerateReadmeTool._dedup_str_list(ctx.get("github_url") or [])
        has_legacy_signal = bool(
            model_list
            or github_url
            or str(ctx.get("ref_md") or "").strip()
            or str(ctx.get("ref_index_js") or "").strip()
        )

        source_md_url = str(ctx.get("source_md_url") or "").strip()
        source_js_url = str(ctx.get("source_js_url") or "").strip()
        github_md_folder_url = str(ctx.get("github_md_folder_url") or "").strip()
        github_js_folder_url = str(ctx.get("github_js_folder_url") or "").strip()
        source_urls = remote_payload.get("source_urls") or {}
        has_source_signal = bool(
            source_md_url
            or source_js_url
            or github_md_folder_url
            or github_js_folder_url
            or str(source_urls.get("md") or "").strip()
            or str(source_urls.get("js") or "").strip()
            or len(ctx.get("source_md_files") or []) > 0
            or len(ctx.get("source_js_files") or []) > 0
            or remote_mode == "url_source"
        )

        # Strong explicit branches first.
        if explicit_mode == "reference" and has_legacy_signal and not has_source_signal:
            return ("reference", "explicit")
        if is_url_source_mode(explicit_mode) and has_source_signal and not has_legacy_signal:
            return ("web_sources", "explicit")

        # Strong data-driven branches.
        if has_source_signal and not has_legacy_signal:
            return ("web_sources", "rule")
        if has_legacy_signal and not has_source_signal:
            return ("reference", "rule")

        # Ambiguous/conflicting: let LLM arbitrate.
        judged = GenerateReadmeTool._llm_judge_generation_mode(ctx)
        if judged in {"reference", "web_sources"}:
            return (judged, "llm")

        # Conservative fallback.
        if explicit_mode == "reference":
            return ("reference", "fallback")
        if is_url_source_mode(explicit_mode):
            return ("web_sources", "fallback")
        return ("reference", "fallback")

    @staticmethod
    def _inject_intel_cpu_xeon_section(
        family_md: str,
        family_index_js: str,
        model_list: List[str],
        model_id_list: List[str],
        mode: str,
    ) -> tuple[str, str]:
        """Safety-net: if the primary generation is missing Intel/Xeon content, augment via LLM
        (MD) + code patch (JS). Only fires when Intel content is absent.
        Falls back to hardcoded append only if the LLM call itself fails.
        """
        if not is_url_source_mode(mode):
            return (family_md, family_index_js)

        md = str(family_md or "")
        js = str(family_index_js or "")

        # Primary generation already produced Intel + Xeon content.
        # Still run the JS patcher to guarantee all mechanical changes (w8a8, visibleIf, etc.).
        target_models = (
            GenerateReadmeTool._dedup_str_list(model_id_list or [])
            or GenerateReadmeTool._dedup_str_list(model_list or [])
        )
        variant_info = GenerateReadmeTool._classify_models_by_variant(target_models)

        if "intel" in md.lower() and "xeon" in md.lower():
            js = GenerateReadmeTool._patch_js_for_intel_xeon(js, variant_info)
            return (md, js)

        print("[inject_intel_xeon] Intel/Xeon content missing — augmenting via LLM + code patch.")

        intel_models = variant_info["intel_models"]
        intel_quant_variants = variant_info["intel_quant_variants"]
        w8a8_model = next((m for m in target_models if "w8a8" in m.lower()), "")
        example_model = next(
            (m for m in intel_models if "instruct" in m.lower() and "8b" in m.lower()
             and "fp8" not in m.lower() and "w8a8" not in m.lower()),
            intel_models[0] if intel_models else target_models[0] if target_models else "meta-llama/Llama-3.1-8B-Instruct",
        )

        # Build per-variant command examples
        quant_pat = re.compile(
            r"(?:[-._])(fp8|bf16|int8|int4|w8a8|w4a16|awq|gptq|quantized(?:\.[a-z0-9]+)*)",
            flags=re.IGNORECASE,
        )
        cmd_examples: List[str] = []
        seen_quants: set = set()
        for m in intel_models:
            qm = quant_pat.search(m)
            if qm:
                raw_q = qm.group(1).lower()
                quant = raw_q.split(".", 1)[1] if raw_q.startswith("quantized.") else raw_q
            else:
                quant = "bf16"
            if quant in seen_quants:
                continue
            seen_quants.add(quant)
            cmd_examples.append(f"# {quant.upper()}\nsglang serve {m} --tp 1 --device cpu")
        intel_cmd_block = "\n\n".join(cmd_examples)

        prompt = f"""You are a technical documentation writer for LLM serving infrastructure.

The README below covers CUDA/AMD backends only. Your task is to AUGMENT it by adding
Intel CPU (Xeon) sections — do NOT remove or change any existing content.
Return the COMPLETE README with Intel Xeon sections inserted in the right places,
matching the style of the existing CUDA/AMD sections.

### SOURCE family_md (keep every line; insert Intel Xeon sections alongside CUDA/AMD):
{md}

### Intel Xeon models (use EXACT IDs):
{json.dumps(intel_models, ensure_ascii=False)}
### Quantization variants: {json.dumps(intel_quant_variants)}
{"### W8A8 model: " + w8a8_model if w8a8_model else ""}

### Intel Xeon launch examples to include (one per quantization variant):
{intel_cmd_block}

### Also include:
- Dual-socket variant: `sglang serve {example_model} --tp 2 --device cpu`
- Benchmark block for Intel Xeon matching CUDA/AMD benchmark style.

Output ONLY valid JSON (no markdown fences):
{{"family_md": "<complete augmented README>"}}"""

        new_md = md
        try:
            response = GenerateReadmeTool.llm.invoke(prompt)
            cleaned = GenerateReadmeTool._strip_think_blocks(response)
            cleaned_lower = cleaned.lstrip().lower()
            if not (cleaned_lower.startswith("<!doctype") or cleaned_lower.startswith("<html")):
                parsed = json.loads(cleaned)
                lm = str(parsed.get("family_md") or "").strip()
                if lm and len(lm) >= len(md) * 0.8:  # must not be shorter than 80% of source
                    new_md = lm
                else:
                    raise ValueError("LLM returned truncated family_md")
        except Exception as e:
            print(f"[inject_intel_xeon][llm_md_failed] {e} — appending hardcoded Intel section")
            # Hardcoded fallback for MD
            intel_md_lines = [f"#### Xeon\n"]
            intel_md_lines.append("Optimized for Intel Xeon Scalable processors using the `--device cpu` flag.\n")
            for m_ex in cmd_examples:
                header, cmd = m_ex.split("\n", 1)
                quant_label = header.strip("# ")
                intel_md_lines.append(f"**{quant_label} Example:**\n```shell\n{cmd}\n```\n")
            if example_model:
                intel_md_lines.append(
                    f"**Dual-Socket Example:**\n```shell\nsglang serve {example_model} --tp 2 --device cpu\n```\n"
                )
            new_md = md.rstrip() + "\n\n" + "\n".join(intel_md_lines)

        # Always apply code-level JS patch regardless of LLM outcome
        new_js = GenerateReadmeTool._patch_js_for_intel_xeon(js, variant_info)
        return (new_md.strip(), new_js.strip())

    @staticmethod
    def _classify_models_by_variant(model_list: List[str]) -> Dict[str, Any]:
        """Analyse model_list and return structured information about which quantization
        variants are present and which are suitable for Intel CPU (Xeon) deployment.

        Returns a dict with:
          - all_models: full deduplicated list
          - intel_models: ALL models in model_list (caller supplies what Intel can serve)
          - quant_variants: sorted list of quantization tags (e.g. ['bf16','fp8','w8a8'])
          - intel_quant_variants: quantization tags in intel_models (same as quant_variants)
          - model_map: dict keyed by "{size}_{category}_{quant}" -> model_id
          - quant_hardware_map: dict keyed by quant -> {"only_on": [...]} or {"universal": True}
            Derived by inspecting model namespace prefixes:
              - amd/ prefix       → AMD-specific
              - RedHatAI/ / Intel/ / intel-labs/ prefix → Intel-specific
              - everything else   → universal (NV, AMD, Intel all get it)
        """
        quant_pattern = re.compile(
            r"(?:[-._])(fp8|bf16|int8|int4|w8a8|w4a16|awq|gptq|quantized(?:\.[a-z0-9]+)*)",
            flags=re.IGNORECASE,
        )
        size_pattern = re.compile(r"[-._](\d+(?:\.\d+)?[bB])(?:[-._]|$)")
        instruct_pattern = re.compile(r"instruct", re.IGNORECASE)

        # Known namespace → hardware affinity mappings
        INTEL_PREFIXES = {"redhatai", "intel", "intel-labs", "openvino"}
        AMD_PREFIXES = {"amd", "rocm"}

        quant_variants: set = set()
        model_map: Dict[str, str] = {}
        # quant -> set of hardware families that have a model for it
        quant_hardware_families: Dict[str, set] = {}

        for m in model_list:
            size_m = size_pattern.search(m)
            size = size_m.group(1).lower() if size_m else "unknown"
            is_instruct = bool(instruct_pattern.search(m))
            quant_m = quant_pattern.search(m)
            if quant_m:
                raw_q = quant_m.group(1).lower()
                quant = raw_q.split(".", 1)[1] if raw_q.startswith("quantized.") else raw_q
            else:
                quant = "bf16"

            quant_variants.add(quant)
            category = "instruct" if is_instruct else "base"
            key = f"{size}_{category}_{quant}"
            model_map[key] = m

            # Determine hardware affinity from namespace prefix
            namespace = m.split("/")[0].lower() if "/" in m else ""
            if any(namespace.startswith(p) for p in INTEL_PREFIXES):
                hw_family = "xeon"
            elif any(namespace.startswith(p) for p in AMD_PREFIXES):
                hw_family = "amd"
            else:
                hw_family = "universal"

            quant_hardware_families.setdefault(quant, set()).add(hw_family)

        # Derive quant_hardware_map:
        # - If a quant has "universal" → available on all hardware (no disabledWhen needed)
        # - If a quant has only "xeon"  → disable when hardware != 'xeon'
        # - If a quant has only "amd"   → disable when hardware is not AMD
        quant_hardware_map: Dict[str, Any] = {}
        amd_hw_ids = {"mi300x", "mi325x", "mi355x"}
        nvidia_hw_ids = {"h100", "h200", "b200", "a100", "a10", "l40"}
        for quant, families in quant_hardware_families.items():
            if "universal" in families:
                quant_hardware_map[quant] = {"universal": True}
            elif families == {"xeon"}:
                quant_hardware_map[quant] = {
                    "only_on": ["xeon"],
                    "disabled_for_others": True,
                }
            elif families == {"amd"}:
                quant_hardware_map[quant] = {
                    "only_on": list(amd_hw_ids),
                    "disabled_for_others": True,
                }
            else:
                quant_hardware_map[quant] = {"universal": True}

        # All models in model_list are Intel-supported (caller defines what Xeon can serve).
        # Do NOT filter by size — if the user provides 405B in model_list, Xeon can serve it.
        intel_models = list(model_list)

        return {
            "all_models": model_list,
            "intel_models": intel_models,
            "quant_variants": sorted(quant_variants),
            "intel_quant_variants": sorted(quant_variants),
            "model_map": model_map,
            "quant_hardware_map": quant_hardware_map,
        }

    @staticmethod
    def _patch_js_for_intel_xeon(js: str, variant_info: Dict[str, Any]) -> str:
        """Programmatically patch the source index.js to add Intel Xeon support.

        Changes made:
        1. Add 'xeon' hardware item to the hardware options list (if not present).
        2. Make quantization visibleIf also true for Intel Xeon.
        3. Add w8a8 (and fp8 if not present) to quantization items.
        4. Add Intel Xeon branch to generateCommand.
        """
        intel_models: List[str] = variant_info.get("intel_models") or []
        intel_quant_variants: List[str] = variant_info.get("intel_quant_variants") or ["bf16"]
        all_models: List[str] = variant_info.get("all_models") or []
        w8a8_model = next((m for m in all_models if "w8a8" in m.lower()), "")
        example_model = next(
            (m for m in intel_models if "instruct" in m.lower() and "8b" in m.lower()
             and "fp8" not in m.lower() and "w8a8" not in m.lower()),
            intel_models[0] if intel_models else "",
        )

        # 0. Remove incorrect disabledWhen from model-size items (8b/70b/405b) that are
        #    present in intel_models. The caller supplies the exact list — if 405B is in
        #    the list, Xeon supports it; we must not grey it out.
        size_pat0 = re.compile(r"[-._](\d+(?:\.\d+)?[bB])(?:[-._]|$)")
        intel_sizes: set = set()
        for m in intel_models:
            sm = size_pat0.search(m)
            if sm:
                intel_sizes.add(sm.group(1).lower())

        def _remove_disabled_when_from_size_item(m: re.Match) -> str:
            item_str = m.group(0)
            item_id_m = re.search(r"id:\s*'(\d+(?:\.\d+)?[bBmM])'", item_str)
            if not item_id_m:
                return item_str
            item_size = item_id_m.group(1).lower()
            if item_size in intel_sizes and "disabledWhen" in item_str:
                # Strip disabledWhen and disabledReason from size items
                item_str = re.sub(r",?\s*disabledWhen:\s*\([^)]*\)\s*=>[^,}]+", "", item_str)
                item_str = re.sub(r",?\s*disabledReason:\s*'[^']*'", "", item_str)
                item_str = re.sub(r",?\s*disabledReason:\s*\"[^\"]*\"", "", item_str)
            return item_str

        js = re.sub(r"\{\s*id:\s*'\d+(?:\.\d+)?[bBmM]'[^}]*\}", _remove_disabled_when_from_size_item, js)

        # 1. Insert xeon hardware item after last AMD item
        if "id: 'xeon'" not in js and 'id: "xeon"' not in js:
            js = re.sub(
                r"(\{\s*id:\s*'mi355x'[^}]*\})",
                r"\1,\n          { id: 'xeon', label: 'Xeon', default: false, backend: 'intel' }",
                js,
            )

        # 2. Make quantization always visible (remove hardware restrictions in visibleIf).
        #    The old logic only showed quantization for 405B or AMD, missing NV+Xeon.
        #    Use disabledWhen on individual items (step 6) instead of hiding the whole group.
        # Replace any restrictive visibleIf with one that always returns true.
        js = re.sub(
            r"(visibleIf:\s*\(values\)\s*=>\s*\{)([\s\S]*?)(,\s*\n\s*items:)",
            r"\1\n          return true;\n        }\3",
            js,
        )
        # Also handle single-line arrow visibleIf patterns
        js = re.sub(
            r"visibleIf:\s*\(values\)\s*=>\s*[^,\n{]+(?:,|\n)",
            "visibleIf: () => true,\n",
            js,
        )

        # 3. Add w8a8 quantization item (only if the item entry is absent)
        if "id: 'w8a8'" not in js and 'id: "w8a8"' not in js and "w8a8" in intel_quant_variants:
            js = re.sub(
                r"(\{\s*id:\s*'fp8'[^}]*\})",
                r"\1,\n          { id: 'w8a8', label: 'W8A8 (INT8)', default: false }",
                js,
            )

        # 4. Add Intel Xeon branch to generateCommand if not present
        if "device cpu" not in js:
            size_pat = re.compile(r"[-._](\d+(?:\.\d+)?[bB])(?:[-._]|$)")
            intel_by_size: Dict[str, Dict[str, str]] = {}
            for m in intel_models:
                sm = size_pat.search(m)
                sz = sm.group(1).lower() if sm else "unknown"
                is_inst = bool(re.search(r"instruct", m, re.IGNORECASE))
                qm = re.search(r"(?:[-._])(fp8|w8a8|w4a16|awq|gptq|quantized(?:\.[a-z0-9]+)*)", m, re.IGNORECASE)
                if qm:
                    raw_q = qm.group(1).lower()
                    quant = raw_q.split(".", 1)[1] if raw_q.startswith("quantized.") else raw_q
                else:
                    quant = "bf16"
                cat = "instruct" if is_inst else "base"
                intel_by_size.setdefault(sz, {})[f"{cat}_{quant}"] = m

            lookup_lines = ["        // Intel Xeon: quantization- and size-aware model selection"]
            for sz, variants in sorted(intel_by_size.items()):
                for variant_key, mid in sorted(variants.items()):
                    cat_part, quant_part = variant_key.split("_", 1)
                    lookup_lines.append(
                        f"        if (modelsize === '{sz}' && category === '{cat_part}'"
                        f" && (quantization === '{quant_part}' || quantization === '{quant_part.upper()}'))"
                        f" modelId = '{mid}';"
                    )
            if not lookup_lines[1:]:
                lookup_lines.append(f"        if (hardware === 'xeon') modelId = '{example_model}';")
            intel_cmd_lines = "\n".join(lookup_lines)
            intel_branch = (
                f"\n      }} else if (hardware === 'xeon') {{\n"
                f"{intel_cmd_lines}\n"
                f"        const xeonTp = modelsize === '405b' ? 4 : (modelsize === '70b' ? 2 : 1);\n"
                f"        args.push(`--tp ${{xeonTp}}`);\n"
                f"        args.push(`--device cpu`);\n"
            )
            # Insert the Xeon branch after the AMD block (look for the closing of the TP block)
            js = re.sub(
                r"(}\s*else\s*\{\s*\n\s*//\s*NVIDIA GPU TP)([\s\S]*?)(}\s*\n\s*// Build command)",
                lambda m2: m2.group(1) + m2.group(2) + "}" + intel_branch + "\n\n      // Build command",
                js,
                count=1,
            )

        # 5. Patch existing Intel Xeon branch: inject per-quant model lookup if missing
        if w8a8_model and "w8a8" not in js:
            intel_by_size_patch: Dict[str, Dict[str, str]] = {}
            size_pat2 = re.compile(r"[-._](\d+(?:\.\d+)?[bB])(?:[-._]|$)")
            for m in intel_models:
                sm = size_pat2.search(m)
                sz = sm.group(1).lower() if sm else "unknown"
                is_inst = bool(re.search(r"instruct", m, re.IGNORECASE))
                qm2 = re.search(r"(?:[-._])(fp8|w8a8|w4a16|awq|gptq|quantized(?:\.[a-z0-9]+)*)", m, re.IGNORECASE)
                if qm2:
                    raw_q = qm2.group(1).lower()
                    quant = raw_q.split(".", 1)[1] if raw_q.startswith("quantized.") else raw_q
                else:
                    quant = "bf16"
                cat = "instruct" if is_inst else "base"
                intel_by_size_patch.setdefault(sz, {})[f"{cat}_{quant}"] = m

            model_lookup_lines: List[str] = ["        // Intel Xeon quantization-aware model selection"]
            for sz, variants in sorted(intel_by_size_patch.items()):
                for variant_key, mid in sorted(variants.items()):
                    cat_part, quant_part = variant_key.split("_", 1)
                    model_lookup_lines.append(
                        f"        if (modelsize === '{sz}' && category === '{cat_part}'"
                        f" && (quantization === '{quant_part}' || quantization === '{quant_part.upper()}'))"
                        f" modelId = '{mid}';"
                    )
            lookup_block = "\n".join(model_lookup_lines) + "\n"
            js = re.sub(
                r"((?:else\s+if\s*\(\s*(?:intelBackends|'\w*xeon\w*'|hardware\s*===\s*'xeon')[^)]*\)\s*\{|hardware\s*===\s*'xeon'\s*\{))([\s\S]*?)(args\.push\(`--tp)",
                lambda m2: m2.group(1) + "\n" + lookup_block + "        " + m2.group(3),
                js,
                count=1,
            )

        # 6. Apply disabledWhen to quantization items based on quant_hardware_map.
        #    - Universal quants: no disabledWhen (available on all hardware)
        #    - Intel-only quants (e.g. w8a8 from RedHatAI): disable when hardware !== 'xeon'
        #    - AMD-only quants: disable when hardware is not AMD
        #    Do NOT add disabledWhen to model-size items (e.g. 405b) — that was incorrect.
        quant_hardware_map: Dict[str, Any] = variant_info.get("quant_hardware_map") or {}
        amd_hw_set = "['mi300x', 'mi325x', 'mi355x']"
        for quant, hw_info in quant_hardware_map.items():
            if hw_info.get("universal"):
                continue  # no restriction
            only_on = hw_info.get("only_on") or []
            if not only_on:
                continue
            if "xeon" in only_on and len(only_on) == 1:
                disabled_fn = "(values) => values.hardware !== 'xeon'"
                reason = f"'{quant.upper()} is only available on Xeon'"
            elif all(h in {"mi300x", "mi325x", "mi355x"} for h in only_on):
                disabled_fn = f"(values) => !{amd_hw_set}.includes(values.hardware)"
                reason = f"'{quant.upper()} is only available on AMD GPUs'"
            else:
                continue
            # Patch the quant item in the JS items array
            js = re.sub(
                rf"(\{{\s*id:\s*'{quant}'[^}}]*?)(,?\s*\}})",
                lambda m, dfn=disabled_fn, rsn=reason: m.group(0)
                if "disabledWhen" in m.group(0)
                else m.group(1).rstrip() + f", disabledWhen: {dfn}, disabledReason: {rsn}" + " }",
                js,
            )

        # 7. Remove stale `intelCpuBackend` export appended by fallback code (not part of component).
        js = re.sub(
            r"\n*//\s*Intel CPU[^\n]*\nexport const intelCpuBackend[\s\S]*?;\s*$",
            "",
            js,
            flags=re.MULTILINE,
        )

        return js

    @staticmethod
    def _build_llm_prompt(ctx: Dict[str, Any], resolved_mode: str) -> str:
        model_list = GenerateReadmeTool._dedup_str_list(ctx.get("model_list") or [])
        model_id_list = GenerateReadmeTool._dedup_str_list(ctx.get("model_id_list") or [])
        all_models = model_id_list or model_list
        example_model = next(
            (m for m in all_models if "instruct" in m.lower() and "8b" in m.lower()
             and "fp8" not in m.lower() and "w8a8" not in m.lower()),
            all_models[0] if all_models else "<MODEL_ID>",
        )

        if is_url_source_mode(resolved_mode):
            # Use full source text — never truncate when augmenting.
            ref_md = str(ctx.get("ref_md") or "")
            ref_js = str(ctx.get("ref_index_js") or "")
            variant_info = GenerateReadmeTool._classify_models_by_variant(all_models)
            intel_models = variant_info["intel_models"]
            intel_quant_variants = variant_info["intel_quant_variants"]
            w8a8_model = next((m for m in all_models if "w8a8" in m.lower()), "")

            # Build per-variant command examples for the prompt
            quant_pat = re.compile(
                r"(?:[-._])(fp8|bf16|int8|int4|w8a8|w4a16|awq|gptq|quantized(?:\.[a-z0-9]+)*)",
                flags=re.IGNORECASE,
            )
            cmd_examples: List[str] = []
            seen_quants: set = set()
            for m in intel_models:
                qm = quant_pat.search(m)
                if qm:
                    raw_q = qm.group(1).lower()
                    quant = raw_q.split(".", 1)[1] if raw_q.startswith("quantized.") else raw_q
                else:
                    quant = "bf16"
                if quant in seen_quants:
                    continue
                seen_quants.add(quant)
                cmd_examples.append(
                    f"# {quant.upper()}\nsglang serve {m} --tp 1 --device cpu"
                )
            intel_cmd_block = "\n\n".join(cmd_examples)

            return f"""You are a technical documentation writer for LLM serving infrastructure.

Your task is to AUGMENT the two source files below by adding Xeon support.
Do NOT rewrite, restructure, or shorten any existing content.
Return the COMPLETE files — every existing line must be preserved exactly as-is,
with Xeon content inserted at the appropriate places following the same style.

---
### SOURCE family_md (keep every line; only add Xeon sections):
{ref_md}

### SOURCE family_index_jsx (keep every line; only extend for Xeon):
{ref_js}

---
### Xeon models to support (use EXACT IDs):
{json.dumps(intel_models, ensure_ascii=False)}

### Quantization variants for Xeon: {json.dumps(intel_quant_variants)}
{"### W8A8 model ID: " + w8a8_model if w8a8_model else ""}
### Example model: {example_model}

---
### What to ADD to family_md (without removing or changing anything existing):
1. Add a new backend subsection for Xeon in the same style as the CUDA and AMD sections.
   - Refer to this hardware as "Xeon" only (NOT "Intel CPU Xeon", NOT "Intel Xeon (CPU)").
   - Include ALL configuration tips relevant to Xeon (TP scaling across sockets, BF16/FP8/W8A8 variants, etc.).
   - Include one launch command block per quantization variant, using exact model IDs:
{intel_cmd_block}
   - Add `--tp 2 --device cpu` variant for dual-socket.
2. Add a Xeon Configuration Tips subsection (matching the style of NVIDIA/AMD tips) covering:
   - Socket-level TP: single socket --tp 1, dual socket --tp 2
   - BF16 / FP8 / W8A8 quantization guidance for CPU deployment
3. Do NOT add a benchmark block for Xeon — only CUDA/AMD benchmarks should remain.
4. Do NOT change any existing sections, headings, or commands.

### What to ADD/CHANGE in family_index_jsx (without removing any existing code):
1. Add `{{ id: 'xeon', label: 'Xeon', default: false, backend: 'intel' }}` to the hardware items list, after the last AMD item.
2. Set `quantization.visibleIf` to ALWAYS return true (`visibleIf: () => true`).
   Do NOT hide quantization for any hardware type — use `disabledWhen` on individual items instead.
3. Add `{{ id: 'w8a8', label: 'W8A8 (INT8)', default: false, disabledWhen: (values) => values.hardware !== 'xeon', disabledReason: 'W8A8 is only available on Xeon' }}` to the quantization items (after fp8).
4. In `generateCommand`, add a Xeon branch (before or after the AMD branch) that:
   - Sets `--device cpu` and `--tp` based on model size (8B→1, 70B→2, 405B→4).
   - Resolves the correct model ID based on (modelsize, category, quantization), e.g.:
     w8a8 → `{w8a8_model or "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8"}`.
   - Does NOT add `--kv-cache-dtype` (unsupported on CPU).
5. Do NOT add `disabledWhen` to model size items (e.g. 405b) — all sizes in model_list are supported.
6. NVIDIA-specific optimizations (speculative decoding, etc.) must only apply when hardware is NVIDIA,
   NOT when hardware is 'xeon'. Change `if (!isAMD)` to `if (!isAMD && hardware !== 'xeon')`.

---
Output ONLY valid JSON with exactly this schema (no markdown fences):
{{
  "family_md": "<complete augmented README — every original line preserved>",
  "family_index_js": "<complete augmented index.jsx — every original line preserved>"
}}"""

        # Legacy / reference mode — keep existing compact prompt.
        return f"""You are generating family-level deployment docs.
Input context JSON:
{json.dumps(ctx, ensure_ascii=False)}

Output ONLY JSON with schema:
{{
  "family_md": "full markdown",
  "family_index_js": "main index.jsx content",
  "memory_cleanup": {{
    "model_list": ["..."]
  }}
}}

Rules:
1. Adapt ref_md/ref_index_js to current model_list/model_id_list.
2. Must align README and JS model choices to model_list.
3. Return valid JSON only.
"""

    @staticmethod
    def _llm_generate_family_artifacts(ctx: Dict[str, Any]) -> Dict[str, Any]:
        resolved_mode = str(ctx.get("generation_mode") or "reference").strip().lower()
        prompt = GenerateReadmeTool._build_llm_prompt(ctx, resolved_mode)
        print(f"[llm_generate] prompt size: {len(prompt)} chars, mode: {resolved_mode}")
        # Use streaming for url_source mode (large prompt + large response) so we
        # see live token output and never block silently.
        if is_url_source_mode(resolved_mode):
            print("[llm_generate] streaming response (url_source mode) ...")
            response = GenerateReadmeTool.llm.invoke_stream(prompt, timeout=600, print_progress=True)
        else:
            response = GenerateReadmeTool.llm.invoke(prompt, timeout=300)
        cleaned = GenerateReadmeTool._strip_think_blocks(response)
        cleaned_lower = cleaned.lstrip().lower()
        if cleaned_lower.startswith("<!doctype html") or cleaned_lower.startswith("<html") or "ie friendly error message walkround" in cleaned_lower:
            raise RuntimeError("LLM endpoint returned HTML/proxy error page instead of JSON.")
        parsed = json.loads(cleaned)
        md = str(parsed.get("family_md") or "").strip()
        idx = str(parsed.get("family_index_js") or "").strip()
        js_files = parsed.get("family_js_files")
        if not isinstance(js_files, list):
            js_files = []
        js_files = GenerateReadmeTool._normalize_js_files(js_files)
        if not js_files and idx:
            js_files = [{"path": "index.js", "content": idx}]
        if not idx and js_files:
            primary = next((x for x in js_files if x["path"].split("/")[-1] == "index.js"), js_files[0])
            idx = str(primary.get("content") or "")
        if not md or not idx:
            raise ValueError("LLM generation returned empty family_md or family_index_js")
        models = GenerateReadmeTool._dedup_str_list(ctx.get("model_list") or [])
        mids = GenerateReadmeTool._dedup_str_list(ctx.get("model_id_list") or [])
        md, idx = GenerateReadmeTool._normalize_artifacts_to_target_models(md, idx, models, mids)
        # For url_source mode: guarantee JS patches are applied even if LLM missed some.
        if is_url_source_mode(resolved_mode):
            all_models = mids or models
            variant_info = GenerateReadmeTool._classify_models_by_variant(all_models)
            idx = GenerateReadmeTool._patch_js_for_intel_xeon(idx, variant_info)
            if js_files:
                primary_idx = next(
                    (i for i, x in enumerate(js_files) if x.get("path", "").split("/")[-1] == "index.js"),
                    0,
                )
                js_files[primary_idx]["content"] = idx
        return {
            "family_md": md,
            "family_index_js": idx,
            "family_js_files": js_files,
            "memory_cleanup": parsed.get("memory_cleanup") if isinstance(parsed.get("memory_cleanup"), dict) else {},
            "source": "llm",
        }

    @staticmethod
    def _compact_generation_memory() -> Dict[str, int]:
        memory = GenerateReadmeTool.global_memory
        model_list = GenerateReadmeTool._dedup_str_list(memory.memory_retrieve("model_list") or [])
        model_id_list = GenerateReadmeTool._dedup_str_list(memory.memory_retrieve("model_id_list") or [])
        model_url_list = GenerateReadmeTool._dedup_str_list(memory.memory_retrieve("model_url_list") or [])
        github_url = normalize_list(memory.memory_retrieve("github_url") or [], fallback_single_str=True, stringify_items=True)

        if model_list:
            memory.memory_store("model_list", model_list)
        if model_id_list:
            memory.memory_store("model_id_list", model_id_list)
        if model_url_list:
            memory.memory_store("model_url_list", model_url_list)
        if model_list:
            if len(github_url) < len(model_list):
                github_url = github_url + [""] * (len(model_list) - len(github_url))
            elif len(github_url) > len(model_list):
                github_url = github_url[: len(model_list)]
            memory.memory_store("github_url", github_url)

        # Keep reference only as lightweight structure guidance after final artifacts are generated.
        family_md = str(memory.memory_retrieve("family_md") or "")
        family_index_js = str(memory.memory_retrieve("family_index_js") or "")
        if family_md.strip() and family_index_js.strip():
            ref_md = str(memory.memory_retrieve("ref_md") or "")
            ref_js = str(memory.memory_retrieve("ref_index_js") or "")
            if ref_md:
                memory.memory_store("ref_md", GenerateReadmeTool._shrink_reference_text(ref_md))
            if ref_js:
                memory.memory_store("ref_index_js", GenerateReadmeTool._shrink_reference_text(ref_js, head_chars=1200, tail_chars=300))
            rp = memory.memory_retrieve("remote_payload") or {}
            if isinstance(rp, dict) and str(rp.get("generation_mode") or "").strip().lower() == "legacy":
                rp2 = dict(rp)
                rp2["content"] = {"from_memory": True}
                memory.memory_store("remote_payload", rp2)
        return {
            "model_list": len(model_list),
            "model_id_list": len(model_id_list),
            "model_url_list": len(model_url_list),
            "github_url": len(github_url),
        }

    @tool("memory_retrieve_generation_context")
    def memory_retrieve_generation_context():
        """Retrieve generation context from GLOBAL_MEMORY for canonical family_content generation."""
        return {
            "generation_mode": GenerateReadmeTool.global_memory.memory_retrieve("generation_mode") or "reference",
            "remote_payload": GenerateReadmeTool.global_memory.memory_retrieve("remote_payload") or {},
            "github_md_folder_url": GenerateReadmeTool.global_memory.memory_retrieve("github_md_folder_url") or "",
            "github_js_folder_url": GenerateReadmeTool.global_memory.memory_retrieve("github_js_folder_url") or "",
            "source_md_url": GenerateReadmeTool.global_memory.memory_retrieve("source_md_url") or "",
            "source_js_url": GenerateReadmeTool.global_memory.memory_retrieve("source_js_url") or "",
            "model_list": GenerateReadmeTool.global_memory.memory_retrieve("model_list") or [],
            "model_id_list": GenerateReadmeTool.global_memory.memory_retrieve("model_id_list") or [],
            "model_url_list": GenerateReadmeTool.global_memory.memory_retrieve("model_url_list") or [],
            "github_url": GenerateReadmeTool.global_memory.memory_retrieve("github_url") or [],
            "ref_md": GenerateReadmeTool.global_memory.memory_retrieve("ref_md") or "",
            "ref_index_js": GenerateReadmeTool.global_memory.memory_retrieve("ref_index_js") or "",
            "source_md_files": GenerateReadmeTool.global_memory.memory_retrieve("source_md_files") or [],
            "source_js_files": GenerateReadmeTool.global_memory.memory_retrieve("source_js_files") or [],
            "family_md": GenerateReadmeTool.global_memory.memory_retrieve("family_md") or "",
            "family_index_js": GenerateReadmeTool.global_memory.memory_retrieve("family_index_js") or "",
            "family_js_files": GenerateReadmeTool.global_memory.memory_retrieve("family_js_files") or [],
            "family_content": GenerateReadmeTool.global_memory.memory_retrieve("family_content") or "",
        }

    @staticmethod
    def _ensure_source_js_from_ref_md() -> None:
        """If source_js_url is empty but ref_md is set, scan the MDX content for a component
        import statement and derive + fetch the JS source into memory.

        This is called at the start of readme_generation because the MDX content is first
        fetched (stored as ref_md / source_md_files) during _prepare_memory, so the raw text
        is already available here without any extra HTTP round-trip.
        """
        memory = GenerateReadmeTool.global_memory
        if not memory:
            return
        source_js_url = str(memory.memory_retrieve("source_js_url") or "").strip()
        if source_js_url:
            return  # Already set — nothing to do.

        # Use the already-fetched MDX text.
        ref_md = str(memory.memory_retrieve("ref_md") or "").strip()
        if not ref_md:
            md_files = normalize_list(memory.memory_retrieve("source_md_files") or [])
            if md_files and isinstance(md_files[0], dict):
                ref_md = str(md_files[0].get("content") or "").strip()
        if not ref_md:
            return

        # We need to know the GitHub repo coordinates to build the JS URL.
        source_md_url = str(memory.memory_retrieve("source_md_url") or "").strip()
        if not source_md_url:
            return

        # Extract owner/repo/branch from the MD URL (same helper as crew.py).
        try:
            from urllib.parse import urlparse as _urlparse
            parsed = _urlparse(source_md_url)
            parts = [p for p in (parsed.path or "").strip("/").split("/") if p]
            if len(parts) < 4:
                return
            owner, repo, _kind, branch = parts[0], parts[1], parts[2], parts[3]
            md_path = "/".join(parts[4:])
        except Exception:
            return

        # Pattern 1: absolute repo-root path import (e.g. '/src/snippets/.../foo.jsx')
        # In Docusaurus, /src/ is relative to the project root, which is the first
        # directory component of the MDX path (e.g. 'docs_new' for 'docs_new/cookbook/...').
        abs_m = re.search(r"import\s+[^'\"]+from\s+['\"]/(src/[^'\"]+\.jsx?)['\"]", ref_md)
        if abs_m:
            comp_path_rel = abs_m.group(1)  # e.g. src/snippets/autoregressive/llama31-deployment.jsx
            # Prepend the docusaurus root (first component of MDX path)
            md_parts = [p for p in md_path.split("/") if p]
            docs_root = md_parts[0] if len(md_parts) >= 2 else ""
            comp_path_abs = f"{docs_root}/{comp_path_rel}" if docs_root else comp_path_rel
            derived_js_url = f"https://github.com/{owner}/{repo}/blob/{branch}/{comp_path_abs}"
        # Pattern 2: Docusaurus @site import
        elif (site_m := re.search(r"import\s+\w+\s+from\s+['\"]@site/([^'\"]+)['\"]", ref_md)):
            comp_rel = site_m.group(1).strip("/")
            if comp_rel.endswith((".js", ".jsx")):
                derived_js_url = f"https://github.com/{owner}/{repo}/blob/{branch}/{comp_rel}"
            else:
                derived_js_url = f"https://github.com/{owner}/{repo}/blob/{branch}/{comp_rel}/index.jsx"
        else:
            # Pattern 3: relative import
            rel_m = re.search(r"import\s+\w+\s+from\s+['\"](\.\.[^'\"]+)['\"]", ref_md)
            if rel_m:
                rel = rel_m.group(1).strip()
                md_dir = "/".join(md_path.split("/")[:-1])
                parts_r = (md_dir + "/" + rel).split("/")
                resolved: List[str] = []
                for p in parts_r:
                    if p == "..":
                        if resolved:
                            resolved.pop()
                    elif p not in ("", "."):
                        resolved.append(p)
                comp_path = "/".join(resolved)
                ext = "" if comp_path.endswith((".js", ".jsx")) else "/index.jsx"
                derived_js_url = f"https://github.com/{owner}/{repo}/blob/{branch}/{comp_path}{ext}"
            else:
                # Pattern 3: component tag hint
                comp_m = re.search(r"<!--\s*组件引用[：:]\s*(\w+)\s*-->|<(\w+ConfigGenerator)\s*/?>", ref_md)
                if comp_m:
                    comp_name = (comp_m.group(1) or comp_m.group(2) or "").strip()
                    if comp_name:
                        derived_js_url = (
                            f"https://github.com/{owner}/{repo}/blob/{branch}"
                            f"/src/components/autoregressive/{comp_name}/index.js"
                        )
                    else:
                        return
                else:
                    return

        print(f"[_ensure_source_js_from_ref_md] derived source_js_url: {derived_js_url}")
        memory.memory_store("source_js_url", derived_js_url)

        # Fetch the JS file content and store it as ref_index_js / source_js_files.
        # Re-use the same fetch infrastructure from crew.py via a lightweight import.
        try:
            from urllib.request import ProxyHandler, Request, build_opener
            import os as _os
            # Parse the GitHub blob URL to get the raw download URL.
            parts2 = [p for p in derived_js_url.replace("https://github.com/", "").split("/") if p]
            # parts2: [owner, repo, 'blob', branch, ...path]
            if len(parts2) >= 5 and parts2[2] == "blob":
                js_path = "/".join(parts2[4:])
                raw_js_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{js_path}"
            else:
                raw_js_url = derived_js_url

            proxy = (
                _os.getenv("https_proxy", "").strip()
                or _os.getenv("HTTPS_PROXY", "").strip()
                or "http://proxy-dmz.intel.com:912"
            )
            opener = build_opener(ProxyHandler({"http": proxy, "https": proxy}))
            req = Request(raw_js_url, headers={"User-Agent": "readme-generator"}, method="GET")
            with opener.open(req, timeout=30) as resp:
                js_content = resp.read().decode("utf-8", errors="ignore")

            memory.memory_store("ref_index_js", js_content)
            memory.memory_store("source_js_files", [{"path": js_path if 'js_path' in dir() else "index.js", "content": js_content}])
            print(f"[_ensure_source_js_from_ref_md] fetched JS content ({len(js_content)} chars) from {raw_js_url}")
        except Exception as e:
            print(f"[_ensure_source_js_from_ref_md][WARN] could not fetch JS content: {e}")

    # ── Pipeline step tools ────────────────────────────────────────────────────────────────
    # Each tool wraps exactly one logical step so that:
    #   - CrewAI verbose output shows per-step input/output when an agent calls them.
    #   - The bypassed _run_stage path in crew.py can call them one-by-one with prints.

    @tool("step_ensure_source_js_url")
    def step_ensure_source_js_url() -> Dict[str, Any]:
        """[Step 1] Derive source_js_url from the MDX component import in the already-fetched MDX text,
        then fetch and store the JS file content as ref_index_js.
        Only runs when source_js_url is not already set."""
        GenerateReadmeTool._ensure_source_js_from_ref_md()
        source_js_url = str(GenerateReadmeTool.global_memory.memory_retrieve("source_js_url") or "")
        ref_index_js = str(GenerateReadmeTool.global_memory.memory_retrieve("ref_index_js") or "")
        return {
            "source_js_url": source_js_url or "(not set)",
            "ref_index_js_fetched": len(ref_index_js) > 0,
            "ref_index_js_chars": len(ref_index_js),
        }

    @tool("step_resolve_generation_mode")
    def step_resolve_generation_mode() -> Dict[str, Any]:
        """[Step 2] Determine whether to use 'web_sources' or 'reference' generation mode
        based on available context in memory. Stores resolved mode back to memory."""
        ctx = GenerateReadmeTool.memory_retrieve_generation_context.func()
        mode, mode_from = GenerateReadmeTool._resolve_generation_mode(ctx)
        GenerateReadmeTool.global_memory.memory_store("generation_mode", mode)
        return {
            "resolved_mode": mode,
            "decision_source": mode_from,
            "model_list_count": len(GenerateReadmeTool._dedup_str_list(ctx.get("model_list") or [])),
            "has_ref_md": bool(str(ctx.get("ref_md") or "").strip()),
            "has_ref_index_js": bool(str(ctx.get("ref_index_js") or "").strip()),
        }

    @tool("step_classify_models")
    def step_classify_models() -> Dict[str, Any]:
        """[Step 3] Classify models from memory into quantization variants and hardware-specific groups
        (universal / Intel Xeon only / AMD only). Returns a summary of the classification."""
        model_id_list = GenerateReadmeTool._dedup_str_list(
            GenerateReadmeTool.global_memory.memory_retrieve("model_id_list") or []
        )
        model_list = GenerateReadmeTool._dedup_str_list(
            GenerateReadmeTool.global_memory.memory_retrieve("model_list") or []
        )
        all_models = model_id_list or model_list
        if not all_models:
            return {"all_models_count": 0, "quant_variants": [], "quant_hardware_map": {}}
        info = GenerateReadmeTool._classify_models_by_variant(all_models)
        return {
            "all_models_count": len(info["all_models"]),
            "intel_models_count": len(info["intel_models"]),
            "quant_variants": info["quant_variants"],
            "quant_hardware_map": {
                q: ("universal" if v.get("universal") else f"only_on={v.get('only_on')}")
                for q, v in info["quant_hardware_map"].items()
            },
        }

    @tool("step_run_llm_generation")
    def step_run_llm_generation() -> Dict[str, Any]:
        """[Step 4] Call the LLM to generate family_md and family_index_js from the generation context.
        Falls back to reference-based generation if the LLM call fails.
        Stores the raw generated dict to pipeline state for the next step."""
        ctx = GenerateReadmeTool.memory_retrieve_generation_context.func()
        llm_error = ""
        try:
            generated = GenerateReadmeTool._llm_generate_family_artifacts(ctx)
        except Exception as e:
            llm_error = f"{type(e).__name__}: {e}"
            generated = GenerateReadmeTool._fallback_generate_from_reference(ctx)
            generated["llm_error"] = llm_error
        GenerateReadmeTool._pipeline["llm_generation_raw"] = generated
        return {
            "source": generated.get("source", "unknown"),
            "family_md_chars": len(str(generated.get("family_md") or "")),
            "family_index_js_chars": len(str(generated.get("family_index_js") or "")),
            "llm_error": llm_error or "none",
        }

    @tool("step_postprocess_artifacts")
    def step_postprocess_artifacts() -> Dict[str, Any]:
        """[Step 5] Apply post-processing to the LLM-generated artifacts:
        model-name normalization, launch-command injection, MDX component reference fix,
        and Xeon section injection/JS patching.
        Stores the processed artifacts to pipeline state."""
        ctx = GenerateReadmeTool.memory_retrieve_generation_context.func()
        generated = GenerateReadmeTool._pipeline.get("llm_generation_raw") or {}
        if not generated:
            generated = GenerateReadmeTool._fallback_generate_from_reference(ctx)

        family_md = str(generated.get("family_md") or "").strip()
        family_index_js = str(generated.get("family_index_js") or "").strip()
        model_list = GenerateReadmeTool._dedup_str_list(ctx.get("model_list") or [])
        model_id_list = GenerateReadmeTool._dedup_str_list(ctx.get("model_id_list") or [])
        mode = str(ctx.get("generation_mode") or "reference").strip().lower()

        # 5a. Normalize model name references in MD/JS
        family_md, family_index_js = GenerateReadmeTool._normalize_artifacts_to_target_models(
            family_md, family_index_js, model_list, model_id_list,
        )
        # 5b. Ensure launch + benchmark command blocks exist in MD
        family_md = GenerateReadmeTool._ensure_readme_command_content(
            family_md=family_md,
            family_index_js=family_index_js,
            model_list=model_list,
            model_id_list=model_id_list,
        )
        # 5c. Fix MDX component reference placeholders → proper import + tag
        family_md = GenerateReadmeTool._fix_md_component_references(family_md)
        # 5d. Inject Intel Xeon sections (MD augmentation + JS patching)
        family_md, family_index_js = GenerateReadmeTool._inject_intel_cpu_xeon_section(
            family_md=family_md,
            family_index_js=family_index_js,
            model_list=model_list,
            model_id_list=model_id_list,
            mode=mode,
        )
        # 5e. Fix truncated closing tags (e.g. </div missing >) in both artifacts
        family_md = GenerateReadmeTool._fix_unclosed_tags(family_md)
        family_index_js = GenerateReadmeTool._fix_unclosed_tags(family_index_js)
        GenerateReadmeTool._pipeline["postprocessed_artifacts"] = {
            "family_md": family_md,
            "family_index_js": family_index_js,
            "family_js_files": generated.get("family_js_files") or [],
        }
        return {
            "family_md_chars": len(family_md),
            "family_index_js_chars": len(family_index_js),
            "has_intel_xeon": "xeon" in family_md.lower() or "device cpu" in family_index_js.lower(),
            "has_mdx_import": "import " in family_md and "from " in family_md,
        }

    @tool("step_store_final_artifacts")
    def step_store_final_artifacts() -> Dict[str, Any]:
        """[Step 6] Store the final postprocessed family_md and family_index_js to Global Memory
        and compact any redundant intermediate memory entries."""
        processed = GenerateReadmeTool._pipeline.get("postprocessed_artifacts") or {}
        if not processed:
            return {"ok": False, "error": "No postprocessed_artifacts in pipeline — run step_postprocess_artifacts first."}

        family_md = str(processed.get("family_md") or "").strip()
        family_index_js = str(processed.get("family_index_js") or "").strip()
        raw_js_files = processed.get("family_js_files") or []

        js_files = GenerateReadmeTool._normalize_js_files(raw_js_files)
        if not js_files and family_index_js:
            js_files = [{"path": "index.js", "content": family_index_js}]
        elif js_files:
            primary_idx = next(
                (i for i, x in enumerate(js_files) if x.get("path", "").split("/")[-1] == "index.js"),
                0,
            )
            js_files[primary_idx]["content"] = family_index_js

        if len(js_files) > 1:
            store_result = GenerateReadmeTool.memory_store_family_multi_artifacts.func(
                family_md=family_md,
                family_js_files_json=json.dumps(js_files, ensure_ascii=False),
            )
        else:
            store_result = GenerateReadmeTool.memory_store_family_artifacts.func(
                family_md=family_md,
                family_index_js=family_index_js,
            )
        GenerateReadmeTool._compact_generation_memory()
        # Clean up intermediate memory keys
        GenerateReadmeTool._pipeline.clear()
        return {
            "ok": True,
            "family_md_chars": len(family_md),
            "family_index_js_chars": len(family_index_js),
            "js_files_count": len(js_files),
            "store_result": store_result,
        }

    # ── Monolithic convenience tool (kept for backward compat; delegates to steps) ───────

    @tool("memory_generate_and_store_family_artifacts")
    def memory_generate_and_store_family_artifacts() -> Dict[str, Any]:
        """Generate family_md/index.js from ref/source + model_list, store artifacts, and compact redundant memory lists."""
        # Derive source_js_url (and fetch JS content) from the already-fetched MDX text
        # when source_js_url was not explicitly provided.
        GenerateReadmeTool._ensure_source_js_from_ref_md()
        ctx = GenerateReadmeTool.memory_retrieve_generation_context.func()
        resolved_mode, mode_from = GenerateReadmeTool._resolve_generation_mode(ctx)
        ctx["generation_mode"] = resolved_mode
        GenerateReadmeTool.global_memory.memory_store("generation_mode", resolved_mode)
        generated: Dict[str, Any]
        llm_error = ""
        try:
            generated = GenerateReadmeTool._llm_generate_family_artifacts(ctx)
        except Exception as e:
            llm_error = f"{type(e).__name__}: {e}"
            generated = GenerateReadmeTool._fallback_generate_from_reference(ctx)

        def _store_from_generated(g: Dict[str, Any]) -> Dict[str, Any]:
            family_md = str(g.get("family_md") or "").strip()
            family_index_js = str(g.get("family_index_js") or "").strip()
            model_list = GenerateReadmeTool._dedup_str_list(ctx.get("model_list") or [])
            model_id_list = GenerateReadmeTool._dedup_str_list(ctx.get("model_id_list") or [])
            family_md_norm, family_index_js_norm = GenerateReadmeTool._normalize_artifacts_to_target_models(
                family_md,
                family_index_js,
                model_list,
                model_id_list,
            )
            family_md_norm = GenerateReadmeTool._ensure_readme_command_content(
                family_md=family_md_norm,
                family_index_js=family_index_js_norm,
                model_list=model_list,
                model_id_list=model_id_list,
            )
            # Replace <!-- 组件引用：ComponentName --> placeholders with proper MDX imports.
            family_md_norm = GenerateReadmeTool._fix_md_component_references(family_md_norm)
            family_md_norm, family_index_js_norm = GenerateReadmeTool._inject_intel_cpu_xeon_section(
                family_md=family_md_norm,
                family_index_js=family_index_js_norm,
                model_list=model_list,
                model_id_list=model_id_list,
                mode=ctx.get("generation_mode", ""),
            )
            # Fix truncated closing tags (e.g. </div missing >) in both artifacts
            family_md_norm = GenerateReadmeTool._fix_unclosed_tags(family_md_norm)
            family_index_js_norm = GenerateReadmeTool._fix_unclosed_tags(family_index_js_norm)
            js_files = GenerateReadmeTool._normalize_js_files(g.get("family_js_files") or [])
            if not js_files and family_index_js_norm:
                js_files = [{"path": "index.js", "content": family_index_js_norm}]
            elif js_files:
                # Sync primary index.js content with any post-processing changes (e.g. Intel injection).
                primary_idx = next(
                    (i for i, x in enumerate(js_files) if x.get("path", "").split("/")[-1] == "index.js"),
                    0,
                )
                js_files[primary_idx]["content"] = family_index_js_norm
            if len(js_files) > 1:
                return GenerateReadmeTool.memory_store_family_multi_artifacts.func(
                    family_md=family_md_norm,
                    family_js_files_json=json.dumps(js_files, ensure_ascii=False),
                )
            return GenerateReadmeTool.memory_store_family_artifacts.func(
                family_md=family_md_norm,
                family_index_js=family_index_js_norm,
            )

        try:
            store_result = _store_from_generated(generated)
        except Exception:
            generated = GenerateReadmeTool._fallback_generate_from_reference(ctx)
            store_result = _store_from_generated(generated)

        compacted = GenerateReadmeTool._compact_generation_memory()
        debug_info = {
            "resolved_generation_mode": resolved_mode,
            "mode_decision_source": mode_from,
            "generation_source": generated.get("source", "unknown"),
            "llm_error": llm_error,
        }
        GenerateReadmeTool.global_memory.memory_store("readme_generation_debug", debug_info)
        return {
            "ok": True,
            "resolved_generation_mode": resolved_mode,
            "mode_decision_source": mode_from,
            "generation_source": generated.get("source", "unknown"),
            "llm_error": llm_error,
            "store_result": store_result,
            "compacted_lengths": compacted,
        }

    @tool("memory_store_family_content")
    def memory_store_family_content(family_content: str):
        """Store canonical family content (single merged md+js content) into GLOBAL_MEMORY with key "family_content"."""
        content = family_content or ""
        md_text = content
        js_text = ""
        js_match = re.search(r"```javascript\s*([\s\S]*?)```", content, flags=re.IGNORECASE)
        if js_match:
            js_text = js_match.group(1).strip()
            md_text = (content[: js_match.start()] + content[js_match.end() :]).strip()
        GenerateReadmeTool._validate_target_models(md_text, js_text)
        GenerateReadmeTool.global_memory.memory_store("family_content", content)
        GenerateReadmeTool.global_memory.memory_store("family_md", md_text)
        GenerateReadmeTool.global_memory.memory_store("family_index_js", js_text)
        return {"ok": True, "family_md_length": len(md_text), "family_index_js_length": len(js_text)}

    @tool("memory_store_family_artifacts")
    def memory_store_family_artifacts(family_md: str, family_index_js: str):
        """Store family README.md + index.js artifacts and compose canonical family_content."""
        GenerateReadmeTool._validate_target_models(family_md or "", family_index_js or "")
        GenerateReadmeTool.global_memory.memory_store("family_md", family_md or "")
        GenerateReadmeTool.global_memory.memory_store("family_index_js", family_index_js or "")
        js_files = [{"path": "index.js", "content": family_index_js or ""}]
        GenerateReadmeTool.global_memory.memory_store("family_js_files", js_files)
        family_content = GenerateReadmeTool._compose_family_content(
            family_md or "",
            family_index_js or "",
            js_files,
        )
        GenerateReadmeTool.global_memory.memory_store("family_content", family_content)
        return {
            "ok": True,
            "family_md_length": len(family_md or ""),
            "family_index_js_length": len(family_index_js or ""),
            "family_content_length": len(family_content or ""),
        }

    @tool("memory_store_family_multi_artifacts")
    def memory_store_family_multi_artifacts(family_md: str, family_js_files_json: str):
        """Store family README.md plus multiple JS files; keeps index.js as primary for compatibility."""
        js_files_raw = []
        try:
            js_files_raw = family_js_files_json if isinstance(family_js_files_json, list) else re.sub(r"^```json|```$", "", str(family_js_files_json).strip(), flags=re.IGNORECASE).strip()
            if isinstance(js_files_raw, str):
                js_files_raw = js_files_raw or "[]"
                js_files_raw = json.loads(js_files_raw)
        except Exception as e:
            raise ValueError(f"family_js_files_json must be valid JSON list: {e}")

        js_files = GenerateReadmeTool._normalize_js_files(js_files_raw)
        index_item = next((x for x in js_files if x["path"].split("/")[-1] == "index.js"), js_files[0] if js_files else {"content": ""})
        family_index_js = str(index_item.get("content") or "")
        GenerateReadmeTool._validate_target_models(family_md or "", family_index_js or "")

        GenerateReadmeTool.global_memory.memory_store("family_md", family_md or "")
        GenerateReadmeTool.global_memory.memory_store("family_index_js", family_index_js)
        GenerateReadmeTool.global_memory.memory_store("family_js_files", js_files)
        family_content = GenerateReadmeTool._compose_family_content(family_md or "", family_index_js or "", js_files)
        GenerateReadmeTool.global_memory.memory_store("family_content", family_content)
        return {"ok": True, "js_file_count": len(js_files), "family_content_length": len(family_content)}
