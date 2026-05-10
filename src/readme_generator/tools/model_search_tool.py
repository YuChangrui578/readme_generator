import re
import os
import requests
from typing import List
from crewai.tools import tool
import traceback
from .memory_tool import GlobalMemory
from .common_utils import is_url_source_mode, normalize_list
import json
from .input_parser_tool import InternelParserLLM

class HuggingFaceModelClient:
    TRUSTED_ORGS = {
        "meta-llama",
        "qwen",
        "mistralai",
        "google",
        "deepseek-ai",
        "microsoft",
        "tiiuae",
        "redhatai",
        "hugging-quants",
        "neuralmagic",
        "unsloth",
        "intel",
    }
    VARIANT_KEYWORDS = {
        "base": ("-base", "_base", " base"),
        "instruct": ("instruct", "instruction"),
        "thinking": ("thinking", "reasoning"),
        "fp8": ("fp8",),
        "awq": ("awq",),
        "quantized": ("quantized", "int4", "int8", "w8a8", "gptq"),
    }

    def _search(self, query: str, limit: int = 40):
        try:
            params={"search":query,"sort":"downloads","direction":"-1","limit":limit}
            API_URL="https://huggingface.co/api/models"
            hf_proxy = os.getenv("README_GENERATOR_HF_HTTPS_PROXY", "http://proxy.ims.intel.com:911")
            resp=requests.get(
                API_URL,
                params=params,
                timeout=10,
                proxies={"http": hf_proxy, "https": hf_proxy},
            )
            resp.raise_for_status()
            models=resp.json() or []
            if not isinstance(models, list):
                return []
            return models
        except Exception as e:
            print(e)
            traceback.print_exc()
            return []

    @staticmethod
    def _category_key(name: str) -> str:
        text = str(name or "").strip().lower()
        if "/" in text:
            text = text.split("/", 1)[1]
        m = re.search(
            r"(qwen|llama|gemma|mistral|deepseek|phi)\s*[-_ ]?(\d+(?:\.\d+)?)?\s*[-_ ]?(\d+(?:\.\d+)?[bBmM])",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            fam = m.group(1).lower()
            ver = (m.group(2) or "").strip()
            size = (m.group(3) or "").replace(" ", "").lower()
            return f"{fam}{ver}-{size}" if ver else f"{fam}-{size}"
        return text

    @staticmethod
    def _family_size_anchor(name: str) -> str:
        text = str(name or "").strip()
        m = re.search(
            r"(qwen|llama|gemma|mistral|deepseek|phi)\s*[-_ ]?(\d+(?:\.\d+)?)?\s*[-_ ]?(\d+(?:\.\d+)?[bBmM])",
            text,
            flags=re.IGNORECASE,
        )
        if not m:
            return text.lower()
        fam = m.group(1).lower()
        ver = (m.group(2) or "").strip()
        size = (m.group(3) or "").replace(" ", "").lower()
        return f"{fam}{ver}-{size}" if ver else f"{fam}-{size}"

    @staticmethod
    def _compact_token(text: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(text or "").lower())

    @classmethod
    def _anchor_in_model_id(cls, model_id: str, anchor: str) -> bool:
        if not anchor:
            return False
        a = cls._compact_token(anchor)
        m = cls._compact_token(model_id)
        return bool(a and m and a in m)

    @classmethod
    def _variant_tag(cls, model_id: str) -> str:
        lid = model_id.lower()
        for tag, kws in cls.VARIANT_KEYWORDS.items():
            if any(k in lid for k in kws):
                return tag
        return "other"

    @staticmethod
    def _query_key(name: str) -> str:
        text = str(name or "").strip().lower()
        if "/" in text:
            text = text.split("/", 1)[1]
        text = re.sub(r"[\s_]+", "-", text)
        text = re.sub(r"-{2,}", "-", text).strip("-")
        return text

    @classmethod
    def _requested_tags(cls, name: str) -> set[str]:
        text = str(name or "").lower()
        tags = set()
        for tag, kws in cls.VARIANT_KEYWORDS.items():
            if any(k in text for k in kws):
                tags.add(tag)
        return tags

    @classmethod
    def _org_score(cls, model_id: str) -> int:
        org = str(model_id or "").split("/", 1)[0].lower()
        return 5 if org in cls.TRUSTED_ORGS else 0

    @classmethod
    def _candidate_score(cls, model_id: str, anchor: str, query: str, requested_tags: set[str]) -> int:
        lid = str(model_id or "").lower()
        score = 0
        if cls._anchor_in_model_id(model_id, anchor):
            score += 8
        qk = cls._query_key(query)
        if qk and cls._compact_token(qk) in cls._compact_token(model_id):
            score += 6
        score += cls._org_score(model_id)
        cand_tag = cls._variant_tag(model_id)
        if requested_tags and cand_tag in requested_tags:
            score += 10
        # slight penalty to random domain-specific forks when official/trusted exists
        if any(x in lid for x in ("legal", "medical", "finance")) and cls._org_score(model_id) == 0:
            score -= 2
        return score

    def _expand_variants(self, name: str) -> List[tuple[str, str]]:
        anchor = self._family_size_anchor(name)
        requested_tags = self._requested_tags(name)
        queries = [name]
        if anchor and anchor.lower() != str(name).lower():
            queries.append(anchor)

        merged: List[str] = []
        seen = set()
        for q in queries:
            for item in self._search(q, limit=50):
                model_id = str(item.get("modelId") or "").strip()
                if not model_id or model_id.lower() in seen:
                    continue
                seen.add(model_id.lower())
                merged.append(model_id)

        if not merged:
            return []

        # keep only same-family-size neighborhood when possible
        narrowed = [mid for mid in merged if self._anchor_in_model_id(mid, anchor)]
        candidates = narrowed if narrowed else merged
        candidates = sorted(
            candidates,
            key=lambda mid: self._candidate_score(mid, anchor=anchor, query=name, requested_tags=requested_tags),
            reverse=True,
        )

        # when input explicitly asks variants (e.g., FP8/AWQ/Instruct), keep those first
        picked: List[str] = []
        used = set()
        if requested_tags:
            for tag in requested_tags:
                for mid in candidates:
                    if mid.lower() in used:
                        continue
                    if self._variant_tag(mid) == tag:
                        picked.append(mid)
                        used.add(mid.lower())
                        break

        # pick one per major variant type
        used_tags = set()
        for mid in candidates:
            if mid.lower() in used:
                continue
            tag = self._variant_tag(mid)
            if tag in {"base", "instruct", "thinking", "fp8", "awq", "quantized"}:
                if tag in used_tags:
                    continue
                used_tags.add(tag)
                picked.append(mid)
                used.add(mid.lower())
            if len(picked) >= 8:
                break

        if not picked:
            picked = candidates[:5]

        return [(mid, f"https://huggingface.co/{mid}") for mid in picked]

    def search_model(self,name):
        variants = self._expand_variants(name)
        if variants:
            return variants[0]
        return None

    def batch_search(self,names):
        # de-dup by normalized query key only (keep different variants like FP8/AWQ/Instruct).
        dedup_names = []
        seen_names = set()
        for n in names:
            s = str(n or "").strip()
            if not s:
                continue
            k = self._query_key(s)
            if k in seen_names:
                continue
            seen_names.add(k)
            dedup_names.append(s)

        model_names=[]
        model_ids=[]
        model_urls=[]
        seen_model_ids=set()
        for name in dedup_names:
            variants = self._expand_variants(name)
            if not variants:
                continue
            for mid,url in variants:
                mk = str(mid).lower()
                if mk in seen_model_ids:
                    continue
                seen_model_ids.add(mk)
                model_names.append(mid)
                model_ids.append(mid)
                model_urls.append(url)

        return {
            "model_list": model_names,
            "model_id_list":model_ids,
            "model_url_list":model_urls
        }

    def batch_search_aligned(self, names: List[str]):
        # legacy mode: keep one-to-one alignment with user input model list.
        input_names = normalize_list(names, fallback_single_str=True, stringify_items=True)
        aligned_names: List[str] = []
        model_ids: List[str] = []
        model_urls: List[str] = []
        for raw in input_names:
            name = str(raw or "").strip()
            if not name:
                continue
            aligned_names.append(name)
            hit = self.search_model(name)
            if hit:
                mid, url = hit
                model_ids.append(mid)
                model_urls.append(url)
            else:
                model_ids.append("")
                model_urls.append("")
        return {
            "model_list": aligned_names,
            "model_id_list": model_ids,
            "model_url_list": model_urls,
        }

class ModelSearchTool:
    hf_client=HuggingFaceModelClient()
    global_memory = None

    @staticmethod
    def _memory() -> GlobalMemory:
        return ModelSearchTool.global_memory or GlobalMemory()

    @tool("memory_retrieve_model_list")
    def memory_retrieve_model_list():
        """Retrieve model_list from GLOBAL_MEMORY.
        Returns: list of model names to search."""
        memory=ModelSearchTool._memory()
        model_list = normalize_list(memory.memory_retrieve("model_list"), fallback_single_str=True, stringify_items=True)
        out = []
        seen = set()
        for m in model_list:
            s = str(m or "").strip()
            if not s:
                continue
            k = HuggingFaceModelClient._query_key(s)
            if k in seen:
                continue
            seen.add(k)
            out.append(s)
        return out

    @tool("llm_dedup_model_list")
    def llm_dedup_model_list(model_list:List[str]) -> List[str]:
        """Use internal LLM to deduplicate/cluster model names into canonical family-size categories.
        Returns: list of canonical model representative names (one per category)."""
        try:
            names = normalize_list(model_list, fallback_single_str=True, stringify_items=True)
            if not names:
                return []
            prompt = (
                "You are given a JSON array of model identifiers/names. "
                "Group them by semantic family+size (e.g., qwen3-235b), "
                "and return a JSON object with key 'canonical_list' whose value is an array of one representative name per group. "
                "Prefer a short canonical form like 'qwen3-235b' when possible. "
                "Input: " + json.dumps(names, ensure_ascii=False)
            )
            resp = InternelParserLLM.llm.invoke(prompt)
            # try to parse JSON from the response
            text = resp.strip()
            # extract first {...} or [...]
            m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
            if m:
                text = m.group(1)
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "canonical_list" in parsed:
                out = normalize_list(parsed["canonical_list"], fallback_single_str=True, stringify_items=True)
                if out:
                    return out
            if isinstance(parsed, list):
                out = normalize_list(parsed, fallback_single_str=True, stringify_items=True)
                if out:
                    return out
        except Exception:
            pass
        # fallback: rule-based dedup (category_key)
        out = []
        seen = set()
        for n in normalize_list(model_list, fallback_single_str=True, stringify_items=True):
            k = HuggingFaceModelClient._category_key(n)
            if k in seen:
                continue
            seen.add(k)
            out.append(n)
        return out

    @tool("huggingface_model_batch_search")
    def huggingface_model_batch_search(model_name_list:List[str])->List[str]:
        """Perform batch search for models on Hugging Face.
        Input: list of model names
        Returns: dictionary containing model_id_list and model_url_list, with one-to-one index correspondence.
        In url_source/web_sources mode the provided model_list is used directly as model_id_list
        without any HuggingFace search or variant expansion."""
        mode = str(ModelSearchTool._memory().memory_retrieve("generation_mode") or "").strip().lower()
        if is_url_source_mode(mode):
            # url_source mode: the caller already knows the exact model IDs — use them as-is.
            names = normalize_list(model_name_list or [], fallback_single_str=True, stringify_items=True)
            deduped: List[str] = []
            seen: set = set()
            for n in names:
                s = str(n or "").strip()
                if not s:
                    continue
                k = HuggingFaceModelClient._query_key(s)
                if k in seen:
                    continue
                seen.add(k)
                deduped.append(s)
            urls = [f"https://huggingface.co/{m}" for m in deduped]
            result = {
                "model_list": deduped,
                "model_id_list": deduped,
                "model_url_list": urls,
            }
        else:
            # legacy mode: keep strict 1:1 mapping with input models.
            result = ModelSearchTool.hf_client.batch_search_aligned(model_name_list or [])
        enriched_model_list = normalize_list(result.get("model_list"), fallback_single_str=False, stringify_items=True)
        if enriched_model_list:
            ModelSearchTool._memory().memory_store("model_list", enriched_model_list)
        return result
    
    @tool("memory_store_model_search_results")
    def memory_store_model_search_results(model_id_list:List[str],model_url_list:List[str],model_list:List[str]=None):
        """Store model search results (model_id_list and model_url_list) into GLOBAL_MEMORY.
        Inputs: model_id_list, model_url_list
        Returns: success message."""
        memory=ModelSearchTool._memory()
        if model_list:
            dedup_model_list = []
            seen = set()
            for m in model_list:
                s = str(m or "").strip()
                if not s:
                    continue
                k = HuggingFaceModelClient._query_key(s)
                if k in seen:
                    continue
                seen.add(k)
                dedup_model_list.append(s)
            memory.memory_store("model_list", dedup_model_list)
        memory.memory_store("model_id_list",model_id_list)
        memory.memory_store("model_url_list",model_url_list)
        return "Stored model search results successfully"
