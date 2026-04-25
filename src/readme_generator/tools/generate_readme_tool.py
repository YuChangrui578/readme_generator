from crewai.tools import tool
import re


class GenerateReadmeTool:
    global_memory = None

    @staticmethod
    def _compose_family_content(family_md: str, family_index_js: str) -> str:
        md_text = (family_md or "").strip()
        js_text = (family_index_js or "").strip()
        if not md_text and not js_text:
            return ""
        if not js_text:
            return md_text
        if not md_text:
            return f"### index.js\n\n```javascript\n{js_text}\n```"
        return f"{md_text}\n\n---\n\n### index.js\n\n```javascript\n{js_text}\n```"

    @staticmethod
    def _validate_target_models(family_md: str, family_index_js: str) -> None:
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
            raise ValueError("Generated artifacts do not align with input model_list/model_id_list.")

    @tool("memory_retrieve_generation_context")
    def memory_retrieve_generation_context():
        """Retrieve generation context from GLOBAL_MEMORY for canonical family_content generation."""
        return {
            "model_list": GenerateReadmeTool.global_memory.memory_retrieve("model_list") or [],
            "model_id_list": GenerateReadmeTool.global_memory.memory_retrieve("model_id_list") or [],
            "model_url_list": GenerateReadmeTool.global_memory.memory_retrieve("model_url_list") or [],
            "github_url": GenerateReadmeTool.global_memory.memory_retrieve("github_url") or [],
            "ref_md": GenerateReadmeTool.global_memory.memory_retrieve("ref_md") or "",
            "ref_index_js": GenerateReadmeTool.global_memory.memory_retrieve("ref_index_js") or "",
            "family_md": GenerateReadmeTool.global_memory.memory_retrieve("family_md") or "",
            "family_index_js": GenerateReadmeTool.global_memory.memory_retrieve("family_index_js") or "",
            "family_content": GenerateReadmeTool.global_memory.memory_retrieve("family_content") or "",
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
        family_content = GenerateReadmeTool._compose_family_content(family_md or "", family_index_js or "")
        GenerateReadmeTool.global_memory.memory_store("family_content", family_content)
        return {
            "ok": True,
            "family_md_length": len(family_md or ""),
            "family_index_js_length": len(family_index_js or ""),
            "family_content_length": len(family_content or ""),
        }
