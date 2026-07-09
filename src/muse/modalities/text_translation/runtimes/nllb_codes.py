"""ISO 639-1 code -> NLLB-200 FLORES-200 code table.

Static data, no logic. NLLB models (facebook/nllb-200-*) do not accept
plain ISO 639-1 codes as language tokens; they use FLORES-200 codes,
which pair an ISO 639-3-ish language identifier with a script subtag
(e.g. "eng_Latn", "zho_Hans"). This table maps the wire-level ISO
639-1 codes muse's /v1/translate and /translate accept to the
FLORES-200 code the NLLB tokenizer expects.

Coverage: every NLLB-200 language that has an ISO 639-1 two-letter
code. NLLB-200 covers ~200 languages total, but roughly half of those
are low-resource languages with no ISO 639-1 assignment (only ISO
639-3); such languages simply have no entry here, matching the design
note that "wire codes are ISO [639-1] only" -- muse does not expose
FLORES-200 or ISO 639-3 codes at the wire layer in v1.

Several ISO 639-1 codes are macrolanguage codes that FLORES-200 splits
into multiple script/dialect variants (e.g. "ar" covers a dozen Arabic
FLORES-200 entries, "zh" covers Simplified and Traditional Chinese,
"ku" covers Kurmanji and Sorani Kurdish, "ms" covers Malay variants,
"mn" covers several Mongolic languages, "no" covers Bokmal and
Nynorsk, "fa" covers Western/Dari Persian variants, "lv" covers
Latvian variants, "sq" covers Albanian variants, "uz" covers Uzbek
variants). Where FLORES-200 has an unambiguous "standard" variant
(e.g. facebook's own NLLB documentation and demos default to it), that
variant is used as the single mapping target; the alternates are not
independently reachable via the ISO 639-1 wire code in v1.

Source: cross-referenced against the published NLLB-200 / FLORES-200
language list (Costa-jussa et al., "No Language Left Behind", 2022)
and the `facebook/nllb-200-distilled-600M` tokenizer's language list.
"""
from __future__ import annotations


ISO_TO_FLORES: dict[str, str] = {
    "af": "afr_Latn",
    "am": "amh_Ethi",
    "ar": "arb_Arab",
    "as": "asm_Beng",
    "ay": "ayr_Latn",
    "az": "azj_Latn",
    "ba": "bak_Cyrl",
    "be": "bel_Cyrl",
    "bg": "bul_Cyrl",
    "bn": "ben_Beng",
    "bo": "bod_Tibt",
    "bs": "bos_Latn",
    "ca": "cat_Latn",
    "cs": "ces_Latn",
    "cy": "cym_Latn",
    "da": "dan_Latn",
    "de": "deu_Latn",
    "dz": "dzo_Tibt",
    "ee": "ewe_Latn",
    "el": "ell_Grek",
    "en": "eng_Latn",
    "eo": "epo_Latn",
    "es": "spa_Latn",
    "et": "est_Latn",
    "eu": "eus_Latn",
    "fa": "pes_Arab",
    "ff": "fuv_Latn",
    "fi": "fin_Latn",
    "fj": "fij_Latn",
    "fo": "fao_Latn",
    "fr": "fra_Latn",
    # "fy" (Western Frisian) intentionally has NO entry: real-tokenizer
    # verification against facebook/nllb-200-distilled-600M on 2026-07-09
    # confirmed it ships no "fry_Latn" token. NLLB-200 does not cover
    # Western Frisian; the prior "fy": "fry_Latn" mapping would force an
    # <unk> BOS token and produce garbage output, so it was removed rather
    # than left as a plausible-looking but wrong entry.
    "ga": "gle_Latn",
    "gd": "gla_Latn",
    "gl": "glg_Latn",
    "gn": "grn_Latn",
    "gu": "guj_Gujr",
    "ha": "hau_Latn",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "hr": "hrv_Latn",
    "ht": "hat_Latn",
    "hu": "hun_Latn",
    "hy": "hye_Armn",
    "id": "ind_Latn",
    "ig": "ibo_Latn",
    "is": "isl_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "jv": "jav_Latn",
    "ka": "kat_Geor",
    "kg": "kon_Latn",
    "kk": "kaz_Cyrl",
    "km": "khm_Khmr",
    "kn": "kan_Knda",
    "ko": "kor_Hang",
    "ku": "kmr_Latn",
    "ky": "kir_Cyrl",
    "lb": "ltz_Latn",
    "lg": "lug_Latn",
    "ln": "lin_Latn",
    "lo": "lao_Laoo",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    "mg": "plt_Latn",
    "mi": "mri_Latn",
    "mk": "mkd_Cyrl",
    "ml": "mal_Mlym",
    "mn": "khk_Cyrl",
    "mr": "mar_Deva",
    "ms": "zsm_Latn",
    "mt": "mlt_Latn",
    "my": "mya_Mymr",
    "ne": "npi_Deva",
    "nl": "nld_Latn",
    "no": "nob_Latn",
    "nb": "nob_Latn",
    "nn": "nno_Latn",
    "ns": "nso_Latn",
    "ny": "nya_Latn",
    "oc": "oci_Latn",
    "om": "gaz_Latn",
    "or": "ory_Orya",
    "pa": "pan_Guru",
    "pl": "pol_Latn",
    "ps": "pbt_Arab",
    "pt": "por_Latn",
    "qu": "quy_Latn",
    "rn": "run_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "rw": "kin_Latn",
    "sc": "srd_Latn",
    "sd": "snd_Arab",
    "sg": "sag_Latn",
    "si": "sin_Sinh",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "sm": "smo_Latn",
    "sn": "sna_Latn",
    "so": "som_Latn",
    "sq": "als_Latn",
    "sr": "srp_Cyrl",
    "ss": "ssw_Latn",
    "st": "sot_Latn",
    "su": "sun_Latn",
    "sv": "swe_Latn",
    "sw": "swh_Latn",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "tg": "tgk_Cyrl",
    "th": "tha_Thai",
    "ti": "tir_Ethi",
    "tk": "tuk_Latn",
    "tl": "tgl_Latn",
    "tn": "tsn_Latn",
    "tr": "tur_Latn",
    "tw": "twi_Latn",
    "ug": "uig_Arab",
    "uk": "ukr_Cyrl",
    "ur": "urd_Arab",
    "uz": "uzn_Latn",
    "vi": "vie_Latn",
    "wo": "wol_Latn",
    "xh": "xho_Latn",
    "yi": "ydd_Hebr",
    "yo": "yor_Latn",
    "zh": "zho_Hans",
    "zu": "zul_Latn",
}
