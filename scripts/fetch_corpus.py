"""Fetch the corpus papers' full text from arXiv into eval/corpus/ as plain text.

    python scripts\\fetch_corpus.py [--force]

Fetches arXiv's official HTML rendering rather than PDFs: its markup is
structured, so navigation, references, captions and author blocks can be dropped
precisely with the stdlib parser and no extra dependency. Writes body text only,
without a title header, because the importer passes title and content
separately. Existing files are skipped unless --force; sources.json records each
paper's arXiv id, version, size and licence so the corpus can be reproduced and
audited.

The licence is recorded because the extraction removes it. arXiv prints it in the
page chrome above the body marker, so the parser drops it by design -- correctly,
since it is not paper content -- and the corpus files therefore carry no
attribution at all. Reading it back from the abstract page and writing it into the
manifest is what keeps the record complete; without it, the one fact needed to
decide whether these files may be redistributed would be the one fact the pipeline
discards.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

import httpx

# Derived from __file__ so the result is the same from any cwd.
CORPUS_DIR = Path(__file__).resolve().parents[1] / "eval" / "corpus"


@dataclass(frozen=True)
class Paper:
    """One corpus paper. `slug` doubles as the file name and the import identifier."""

    arxiv_id: str
    slug: str
    title: str


# Two groups, serving opposite purposes.
#
# The first five are the papers the evaluation questions are written against. They have to
# stay clearly distinguishable from each other: a question needs one unambiguous expected
# source, and overlapping papers would make "which of these does the passage come from"
# unanswerable.
#
# The other fifteen are distractors, and are deliberately not distinguishable from the
# first five -- they cover the same ground (efficient attention, dense retrieval, encoder
# pretraining, sentence embeddings) precisely so that they compete for the same queries.
# A corpus holding only the target papers flatters retrieval: with those five documents and
# top_k=5, a ranking drawn at random finds the expected source about 0.68 of the time, so a
# perfect hit rate means almost nothing. With all twenty the same figure is 0.188, which is
# what lets the hit rate be read at all. No question is written against a distractor, so
# these only have to be plausible.
#
# Dropped while selecting, and why: Reformer (2000.04487) has no HTML rendering; ANCE
# (2010.02625) extracts to under 10,000 characters, below the floor that marks a broken
# body boundary; SPLADE v2 (2109.10086) is CC BY-NC-SA, and a non-commercial restriction is
# one nothing else in the corpus carries; GPT-3 and CLIP are three to four times the length
# of everything else and further from the rest topically.
PAPERS: list[Paper] = [
    Paper("1706.03762", "attention-is-all-you-need", "Attention Is All You Need"),
    Paper("1810.04805", "bert", "BERT: Pre-training of Deep Bidirectional Transformers"),
    Paper("2005.11401", "retrieval-augmented-generation", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"),
    Paper("2004.04906", "dense-passage-retrieval", "Dense Passage Retrieval for Open-Domain Question Answering"),
    Paper("1908.10084", "sentence-bert", "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"),
    # Efficient and long-sequence attention.
    Paper("2004.05150", "longformer", "Longformer: The Long-Document Transformer"),
    Paper("2007.14062", "big-bird", "Big Bird: Transformers for Longer Sequences"),
    Paper("2205.14135", "flash-attention", "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"),
    # Dense retrieval and retrieval-augmented generation.
    Paper("2004.12832", "colbert", "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT"),
    Paper("2002.08909", "realm", "REALM: Retrieval-Augmented Language Model Pre-Training"),
    Paper("2007.01282", "fusion-in-decoder", "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering"),
    Paper("2010.08191", "rocketqa", "RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering"),
    Paper("2112.09118", "unsupervised-dense-retrieval", "Unsupervised Dense Information Retrieval with Contrastive Learning"),
    # Encoder pretraining.
    Paper("1910.01108", "distilbert", "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"),
    Paper("1909.11942", "albert", "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations"),
    Paper("2003.10555", "electra", "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators"),
    Paper("1907.11692", "roberta", "RoBERTa: A Robustly Optimized BERT Pretraining Approach"),
    # Sentence embeddings and adaptation.
    Paper("2104.08821", "simcse", "SimCSE: Simple Contrastive Learning of Sentence Embeddings"),
    Paper("2106.09685", "lora", "LoRA: Low-Rank Adaptation of Large Language Models"),
    Paper("1910.10683", "t5", "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"),
]


class ArxivHtmlTextExtractor(HTMLParser):
    """Extract the body text of an arXiv HTML page.

    This selects content rather than stripping tags: arXiv's markup separates the
    body from page furniture, so the two can be told apart exactly.

    Collection starts at the document-title element and stays off until then,
    because the top of the page carries feedback forms, banners and license text.
    `text()` raises if that marker never appears instead of falling back to
    collecting from the top, which would silently index license and navigation
    text as if it were the paper.

    MathML internals are dropped (they flatten into glyph fragments), but each
    `<math>` element's `alttext` -- its LaTeX source -- is put back inline, so
    sentences such as "a neighborhood of size $r$ in" do not lose their content.

    Everything from the bibliography section onward is dropped: it is long, is
    almost never the answer to a question, and contains every term used in the
    paper, so it would score well for nearly any query.
    """

    # Dropped together with their contents. `math` is absent on purpose: its
    # contents go, but the alttext on its start tag is kept, so it has its own
    # branch in handle_starttag.
    DROP_TAGS = {"script", "style", "nav", "header", "footer", "svg", "figure", "table", "cite"}

    # Elements with one of these classes are dropped with their contents; matched
    # on whole class tokens, never as substrings -- "ltx_document ltx_authors_1line"
    # also contains the substring "ltx_authors" and would take the whole article
    # down with it.
    DROP_CLASSES = {"ltx_authors"}

    # Emit a line break around these so paragraphs stay separated.
    BLOCK_TAGS = {"p", "div", "section", "h1", "h2", "h3", "h4", "h5", "h6", "li", "ul", "ol", "br"}

    # Marker for the start of the body; nothing before it is collected.
    START_CLASS = "ltx_title_document"

    def __init__(self) -> None:
        # convert_charrefs (the default) resolves entities such as &amp; or &#x2019;
        # into characters, instead of leaving them in the text.
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        # Names of the tags still open, in nesting order.
        self._open_tags: list[str] = []
        # Length `_open_tags` must fall below to leave the dropping state; None
        # means not dropping.
        #
        # Depth is recorded rather than a stack of dropped tag names because a
        # dropped element containing a tag of the same name would otherwise close
        # the drop early and leak that element's text into the corpus -- silently,
        # with the output merely shorter. Only a genuinely closed subtree lowers
        # the depth.
        self._drop_until: int | None = None
        # Set once the bibliography is reached; nothing is collected after that.
        self._in_bibliography = False
        # Whether the body start marker has been seen.
        self._started = False

    @classmethod
    def _is_droppable(cls, tag: str, attrs: list[tuple[str, str | None]]) -> bool:
        if tag in cls.DROP_TAGS:
            return True
        classes = set((dict(attrs).get("class") or "").split())
        return bool(cls.DROP_CLASSES & classes)

    def _pop_tag(self, tag: str) -> None:
        """Pop a closing tag, discarding any tags left open above it.

        arXiv's HTML is occasionally mis-nested (`<figure><svg></figure></svg>`).
        Insisting on a matching top of stack would leave the drop state stuck
        forever and truncate the rest of the document, with nothing but a shorter
        output to show for it.
        """
        if not self._open_tags:
            return
        if self._open_tags[-1] == tag:
            self._open_tags.pop()
            return
        if tag in self._open_tags:
            while self._open_tags and self._open_tags.pop() != tag:
                pass

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._in_bibliography:
            return
        self._open_tags.append(tag)

        if self._drop_until is not None:
            # Already dropping: the recorded depth covers the whole subtree.
            return

        if tag == "section" and dict(attrs).get("id") == "bib":
            self._in_bibliography = True
            return

        if self._is_droppable(tag, attrs):
            self._drop_until = len(self._open_tags)
            return

        if self.START_CLASS in set((dict(attrs).get("class") or "").split()):
            self._started = True

        if not self._started:
            return

        if tag == "math":
            # Drop the MathML internals but keep the alttext (the LaTeX source) on
            # the start tag, so "of size $r$ in" does not become "of size in". The
            # surrounding spaces stop it fusing with neighbouring words; the extras
            # are collapsed by _normalize_whitespace.
            #
            # _drop_until is reused to skip the MathML internals rather than adding
            # another flag: "drop until this depth closes" is exactly what is needed.
            alttext = dict(attrs).get("alttext")
            if alttext:
                self._parts.append(f" {_latex_to_text(alttext)} ")
            self._drop_until = len(self._open_tags)
            return

        if tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._in_bibliography:
            return

        if self._drop_until is not None:
            self._pop_tag(tag)
            if len(self._open_tags) < self._drop_until:
                self._drop_until = None
            return

        self._pop_tag(tag)

        if not self._started:
            return

        if tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Handle self-closing tags such as `<br/>` or `<path/>`.

        Handled separately because the default implementation calls
        handle_starttag and handle_endtag in turn, and a block tag emits a newline
        in both -- `<br/>` would become two, leaving a blank line mid-paragraph.
        """
        if self._in_bibliography or self._drop_until is not None or not self._started:
            return
        if tag == "br":
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_bibliography or self._drop_until is not None or not self._started:
            return
        self._parts.append(data)

    def text(self) -> str:
        if not self._started:
            raise ValueError(
                f"Body start marker (element with class {self.START_CLASS!r}) not found. "
                "arXiv's HTML structure may have changed -- open the page and check; do not skip "
                "this check, or the page's license notice and navigation text will be stored as "
                "paper content."
            )
        return _normalize_whitespace("".join(self._parts))


# LaTeX commands that are pure layout wrapping: drop the command, keep what is in
# the braces (\mathrm{softmax} -> softmax, \textsc{BASE} -> BASE). Run repeatedly
# so that one level of nesting is unwrapped too (\mathbf{\mathrm{x}} -> x).
_LATEX_WRAPPERS = re.compile(
    r"\\(?:mathrm|mathbf|mathit|mathsf|mathtt|mathcal|mathbb|mathfrak|mathscr"
    r"|text|textrm|textit|textbf|textsc|textsf|texttt|textnormal"
    r"|operatorname|mbox"
    # Accents: \hat{x} -> x. Loses the "this is an estimate", but keeps a stray
    # accent command out of the middle of a sentence.
    r"|hat|tilde|bar|vec|dot|ddot|check|breve)\s*\{([^{}]*)\}"
)

# Common symbols mapped to Unicode: readable, and it keeps the tokenizer from
# splitting out rare fragments such as "\theta".
#
# Only parameterless commands are listed. Forms such as \frac{1}{\sqrt{d_k}},
# which take arguments containing further braces, are left alone -- see
# _latex_to_text for why.
_LATEX_SYMBOLS = {
    # operators
    "cdot": "·", "times": "×", "div": "÷", "pm": "±", "mp": "∓",
    "geq": "≥", "leq": "≤", "neq": "≠", "approx": "≈", "equiv": "≡",
    "ll": "≪", "gg": "≫", "propto": "∝", "infty": "∞",
    # sets and logic
    "in": "∈", "notin": "∉", "subset": "⊂", "supset": "⊃",
    "cup": "∪", "cap": "∩", "emptyset": "∅",
    "forall": "∀", "exists": "∃", "neg": "¬", "land": "∧", "lor": "∨",
    # arrows
    "rightarrow": "→", "to": "→", "leftarrow": "←",
    "Rightarrow": "⇒", "Leftarrow": "⇐", "leftrightarrow": "↔",
    # ellipses and delimiters
    "cdots": "···", "ldots": "…", "dots": "…",
    "langle": "⟨", "rangle": "⟩", "prime": "′",
    # big operators
    "sum": "Σ", "prod": "Π", "int": "∫",
    # Greek letters
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ", "eta": "η",
    "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π",
    "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ",
    "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
    "Pi": "Π", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
    # Author macros that LaTeXML leaves unexpanded in the source, e.g. the
    # \mathbb{R} shorthands used throughout these papers.
    "inR": "R", "inN": "N", "inZ": "Z", "inC": "C", "inQ": "Q",
}

# The alternation is ordered by descending length: regex alternation takes the
# first match, so listing `in` before `inR` would turn \inR into `in` + R.
#
# It ends with `(?![a-zA-Z])` rather than a word boundary, because a word boundary
# treats `_` as a word character and would therefore miss \beta_{1} -- the most
# common form in formulas. Looking only for a following letter still rejects
# \thetaX.
_LATEX_SYMBOL_RE = re.compile(
    r"\\(" + "|".join(sorted(_LATEX_SYMBOLS, key=len, reverse=True)) + r")(?![a-zA-Z])"
)

# Whitespace-only commands: \, \; \: \! \quad \qquad \hspace{...} -> one space
_LATEX_SPACES = re.compile(r"\\[.,;:!>]|\\(?:quad|qquad)\b|\\hspace\*?\{[^{}]*\}")

# Delimiter sizing commands, no semantic content: \left( \right) \big[ ...
_LATEX_DELIMITERS = re.compile(r"\\(?:left|right|middle|big|Big|bigg|Bigg)\b")

# Escapes: LaTeX writes these characters as \% \_ \& \# \$ \{ \}, restored here
_LATEX_ESCAPES = re.compile(r"\\([%$#&_{}])")


def _latex_to_text(latex: str) -> str:
    """Turn a formula's LaTeX source into text a reader can follow and a search can hit.

    The goal is not correct rendering: \\mathrm{softmax}(QK^{T}) simply becomes
    softmax(QK^{T}), so a reader sees the softmax and a search for it matches.

    Only single-level, non-nested replacements are made. Something like
    \\frac{1}{\\sqrt{d_{k}}} is left as it is, not because it is hard to parse but
    because half-cleaned LaTeX is worse than untouched LaTeX: a simple
    \\frac{a}{b} rewritten to a/b while nested ones are not would leave two
    spellings of the same construct in the corpus, with no single rule to search
    for when something looks wrong. Rendering formulas faithfully needs a MathML
    renderer, not more regexes here -- extracting the body text is this script's
    only promise.
    """
    text = latex

    # Bounded rather than `while True`: LaTeX can in principle be written so the
    # replacements never converge, and a corpus-fetching script should not risk
    # looping forever over that.
    for _ in range(5):
        replaced = _LATEX_WRAPPERS.sub(r"\1", text)
        if replaced == text:
            break
        text = replaced

    text = _LATEX_SYMBOL_RE.sub(lambda m: _LATEX_SYMBOLS[m.group(1)], text)
    text = _LATEX_DELIMITERS.sub("", text)
    text = _LATEX_SPACES.sub(" ", text)
    text = _LATEX_ESCAPES.sub(r"\1", text)

    # Collapse whitespace once more: replaced space commands stack with the spaces
    # already in the formula. Runs of whitespace inside a formula mean nothing.
    return re.sub(r"\s+", " ", text).strip()


def _normalize_whitespace(raw: str) -> str:
    """Tidy up the extracted text: collapse runs of whitespace, remove spaces left
    before punctuation by stripped formulas and citation markers, and reduce runs
    of blank lines to one so paragraphs are separated by a single blank line.

    Collapsing also catches non-breaking spaces, which `\\s` matches and arXiv's
    HTML uses liberally; left in, they reach the stored chunk text and skew both
    token counts and embeddings.
    """
    lines = []
    for line in raw.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        # Only trailing punctuation: a space before an opening bracket is correct.
        line = re.sub(r"\s+([,.;:!?%)\]}])", r"\1", line)
        lines.append(line)

    kept: list[str] = []
    previous_blank = True  # no leading blank line
    for line in lines:
        if line:
            kept.append(line)
            previous_blank = False
        elif not previous_blank:
            kept.append("")
            previous_blank = True

    return "\n".join(kept).strip()


# The version watermark arXiv prints in the page corner:
#     arXiv:1706.03762v7 [cs.CL] 02 Aug 2023
# It is the only record of which revision was fetched: /html/<id> carries no
# version and is not redirected to one, so arXiv just serves the latest.
_WATERMARK = re.compile(r'<div id="watermark-tr">\s*([^<]+?)\s*</div>')

# Below this many characters, extraction is considered to have failed: these
# papers all run well past 20k, so hitting this means the body boundary no longer
# works and only a fraction of the text was collected.
MIN_PLAUSIBLE_CHARS = 10_000


# The licence link on the abstract page, matched on its href rather than on its link text
# or its surrounding element, because arXiv renders the two kinds differently:
#
#   non-exclusive: <div class="abs-license"><a href="http://arxiv.org/licenses/..." >view license</a></div>
#   Creative Commons: <div class="abs-license"><a href="http://creativecommons.org/..." >
#                     <img .../> <span>view license</span></a></div>
#
# The CC form wraps the text in an <img> and a <span>, so a pattern expecting "view license"
# to be the link's immediate content matches the non-exclusive licence and silently misses
# both CC licences. That is the worst possible way round: the papers it drops are the ones
# whose terms are strictest, and an empty value cannot be told from a licence that was never
# read. Matching the href describes the thing actually wanted -- a URL that is a licence URL.
#
# The footer's "Copyright" link does not match: it points at info.arxiv.org/help/license/,
# and the pattern requires the arxiv.org/licenses/ path that the licence itself lives at.
_LICENSE = re.compile(
    r'href="([^"]*(?:arxiv\.org/licenses/|creativecommons\.org/licenses/)[^"]*)"'
)


def extract_version(html: str) -> str:
    """Return the version watermark, or an empty string if the page has none."""
    match = _WATERMARK.search(html)
    return match.group(1) if match else ""


def extract_license(html: str) -> str:
    """Return the licence URL from an arXiv abstract page, or "" if there is no link."""
    match = _LICENSE.search(html)
    return match.group(1) if match else ""


def extract_text(html: str) -> str:
    """Extract the body text from an arXiv HTML page; raises if no body is found."""
    parser = ArxivHtmlTextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()


def fetch_paper(client: httpx.Client, paper: Paper) -> tuple[str, str, str]:
    """Download one paper; returns (body text, version watermark, requested URL)."""
    url = f"https://arxiv.org/html/{paper.arxiv_id}"
    response = client.get(url)
    response.raise_for_status()

    text = extract_text(response.text)

    # Fail rather than write a file that looks fine but holds almost no text. Such
    # a file breaks nothing immediately; it shows up much later as inexplicably
    # poor retrieval, sending you after the retrieval code instead of the corpus.
    if len(text) < MIN_PLAUSIBLE_CHARS:
        raise ValueError(
            f"Extracted only {len(text):,} characters (expected >{MIN_PLAUSIBLE_CHARS:,}) -- "
            "the body boundary may have broken; check arXiv's HTML structure"
        )

    return text, extract_version(response.text), url


def fetch_license(client: httpx.Client, paper: Paper) -> str:
    """Read one paper's licence URL from its abstract page, or "" if it cannot be read.

    A second request to a different path: the body is on /html/ and the licence only on
    /abs/. Never raises. The manifest is a record, and a paper whose licence could not be
    read is still worth fetching and still worth having in the corpus; losing the whole
    paper over a metadata request would trade a usable corpus for a complete manifest.

    The empty string is the one case where the file cannot speak for itself: it means the
    request failed or the page carried no link, and never that the paper has no licence.
    Every arXiv paper has one, so an empty value here says the record is incomplete rather
    than that there is nothing to record.
    """
    try:
        response = client.get(f"https://arxiv.org/abs/{paper.arxiv_id}")
        response.raise_for_status()
    except httpx.HTTPError:
        return ""
    return extract_license(response.text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch corpus paper full text from arXiv")
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-fetch files that already exist (skipped by default, which keeps the run idempotent)",
    )
    args = parser.parse_args()

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = CORPUS_DIR / "sources.json"

    # Skipped files are not downloaded again, so they carry no version -- inherit it
    # from the previous manifest instead of blanking it on every rerun.
    previous = _load_previous_entries(manifest_path)

    # One client reuses the connection across papers; the User-Agent identifies the
    # script, since arXiv rate-limits anonymous traffic.
    headers = {"User-Agent": "ai-paper-qa/0.1 (learning project; corpus fetch script)"}
    sources = []
    failures = 0

    with httpx.Client(headers=headers, follow_redirects=True, timeout=60.0) as client:
        for paper in PAPERS:
            path = CORPUS_DIR / f"{paper.slug}.txt"

            if path.exists() and not args.force:
                remembered = previous.get(paper.slug, {})
                # Backfilled rather than only inherited. The licence was added to the
                # manifest after these files were first fetched, so inheriting alone would
                # record every already-present paper as unlicensed -- permanently, since
                # the empty value never looks like it needs filling. One metadata request
                # per missing entry, and the .txt files are still never rewritten.
                license_url = remembered.get("license_url") or fetch_license(client, paper)
                entry = _source_entry(
                    paper,
                    path,
                    remembered.get("version", ""),
                    license_url,
                )
                print(f"  skipped  {paper.slug:32} already present ({entry['characters']:,} characters)")
                sources.append(entry)
                continue

            try:
                text, version, url = fetch_paper(client, paper)
            except Exception as error:  # noqa: BLE001 -- report the failure, keep going
                print(f"  failed  {paper.slug:32} {type(error).__name__}: {error}")
                failures += 1
                continue

            license_url = fetch_license(client, paper)

            # Explicit UTF-8: the text is full of non-ASCII (Greek letters, dashes,
            # curly quotes) that the Windows default codec would reject on write.
            path.write_text(text, encoding="utf-8")
            entry = _source_entry(paper, path, version, license_url)
            print(f"  done  {paper.slug:32} {len(text):>7,} characters  {version}")
            sources.append(entry)

    # sources.json records each paper's origin, version and size alongside the .txt
    # files, so "which text was this measured on" has an answer.
    manifest = {
        "generated_by": "scripts/fetch_corpus.py",
        "note": (
            "Body text comes from arXiv's official HTML version. References, author blocks, page "
            "navigation, figure captions and tables were removed; formulas keep their LaTeX source "
            "(lightly cleaned up) rather than being rendered as math. Each paper's licence is "
            "recorded in license_url, read from its abstract page: the extraction drops the "
            "licence notice from the body, so this field is the corpus's only attribution record."
        ),
        "papers": sources,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\nCorpus directory: {CORPUS_DIR}")
    if not sources:
        print("⚠️  No paper was fetched successfully.")
        return 1
    if failures:
        print(f"⚠️  {failures} papers failed to fetch, corpus incomplete -- results unreliable, please rerun.")
        return 1
    return 0


def _load_previous_entries(manifest_path: Path) -> dict[str, dict]:
    """Read the previous sources.json keyed by slug; {} if it is missing or corrupt.

    Deliberately does not raise: the manifest is a record, and losing it must not
    block re-fetching. The opposite of the body-extraction logic, whose failures
    are loud because its output is used as data.
    """
    if not manifest_path.exists():
        return {}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {entry["slug"]: entry for entry in data.get("papers", [])}
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}


def _source_entry(paper: Paper, path: Path, version: str, license_url: str) -> dict:
    return {
        "slug": paper.slug,
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "url": f"https://arxiv.org/abs/{paper.arxiv_id}",
        "version": version,
        "license_url": license_url,
        "file": path.name,
        "characters": len(path.read_text(encoding="utf-8")),
    }


if __name__ == "__main__":
    sys.exit(main())
