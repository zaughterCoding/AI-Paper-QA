"""Tests for the corpus fetching script.

Unlike the other scripts, this one produces data: its output is loaded into the vector
store and later feeds evaluation, so a bad run leaves a file that looks fine but is short
or full of navigation text, with no error to point at. These tests pin the extraction
contracts: what must be dropped, what must be kept, and when the script must fail loudly.

`_latex_to_text` and `_normalize_whitespace` are private, and are tested directly on
purpose: they have many edge cases, and covering them through `extract_text` would mean
hand-building an HTML page per case. They are pure functions with clear contracts.
"""

import json
import re
from pathlib import Path

import pytest

from scripts.fetch_corpus import (
    CORPUS_DIR,
    PAPERS,
    _latex_to_text,
    _normalize_whitespace,
    extract_license,
    extract_text,
    extract_version,
    fetch_paper,
)


_DEFAULT_AUTHORS = (
    '<span class="ltx_role_author">A. Author</span>'
    '<span class="ltx_note">thanks: equal contribution, listing order is random</span>'
)


def _page(*body_parts: str, authors_html: str = _DEFAULT_AUTHORS) -> str:
    """Build HTML shaped like a real arXiv page.

    The real class names and nesting are kept deliberately: the tests are only worth
    anything if they go red when that markup changes, which a simplified fake would not.
    """
    return (
        "<html><body>"
        # page chrome before the body, all of which must be filtered out
        '<div id="infobox">'
        '<a href="https://info.arxiv.org/help/license">License: arXiv.org perpetual</a>'
        '<div id="watermark-tr">arXiv:1234.56789v2 [cs.CL] 01 Jan 2024</div>'
        "</div>"
        # The article class contains "ltx_authors_1line", which is not the author block;
        # matching on the "ltx_authors" substring would discard the whole article.
        '<div class="ltx_page_content">'
        '<article class="ltx_document ltx_authors_1line">'
        '<p class="ltx_p">Provided proper attribution is provided.</p>'
        '<h1 class="ltx_title ltx_title_document">A Paper</h1>'
        f'<div class="ltx_authors">{authors_html}</div>' + "".join(body_parts) + '<section id="bib" class="ltx_bibliography">'
        "<h2>References</h2>"
        "<ul><li>A cited paper that must not leak into the corpus. 2020.</li></ul>"
        "</section></article></div></body></html>"
    )


# --- where the body starts and ends -------------------------------------------


def test_keeps_the_body() -> None:
    text = extract_text(_page('<p class="ltx_p">The Transformer is a model.</p>'))

    assert "The Transformer is a model." in text


def test_ignores_page_chrome_before_the_title() -> None:
    """The licence banner and version watermark must not reach the corpus; nothing would
    report them, they would just attach stray text to every chunk."""
    text = extract_text(_page('<p class="ltx_p">Body.</p>'))

    assert "License" not in text
    assert "infobox" not in text


def test_ignores_the_author_block() -> None:
    """The author block is dropped: its footnotes are contribution statements, not
    paper content."""
    text = extract_text(_page('<p class="ltx_p">Body.</p>'))

    assert "A. Author" not in text
    assert "equal contribution" not in text


def test_ignores_the_author_block_when_it_contains_nested_divs() -> None:
    """A same-name tag nested inside the dropped block must not end the drop.

    Tracking the drop with a stack of tag names, rather than a single flag matched by
    name, is what makes this work: a naive match lets the inner close tag clear the outer
    drop state and leak the author names into the corpus, silently.
    """
    nested = '<div class="ltx_affiliation">Dept.</div><span>A. Author</span>'
    html = _page('<p class="ltx_p">Body.</p>', authors_html=nested)

    text = extract_text(html)

    assert "A. Author" not in text
    assert "Dept." not in text
    assert "Body." in text


def test_does_not_confuse_ltx_authors_1line_with_the_author_block() -> None:
    """Regression: `ltx_authors_1line` is a layout marker, not the author block.

    It sits on the `<article>` element, which wraps the entire paper, so matching the
    "ltx_authors" substring would drop all of the content and return an empty string
    without any error.
    """
    text = extract_text(_page('<p class="ltx_p">This must survive.</p>'))

    assert "This must survive." in text


def test_ignores_the_bibliography() -> None:
    """References are dropped wholesale: they are long, almost never the answer, and
    share every term in the paper, so they score well by accident."""
    text = extract_text(_page('<p class="ltx_p">Body.</p>'))

    assert "cited paper" not in text
    assert "References" not in text


def test_escaped_markup_in_the_source_arrives_as_text() -> None:
    """`&lt;b&gt;` in a paper becomes `<b>` in the extracted body, and is not a defect.

    T5's appendix reproduces web documents from C4, which carry their own markup. arXiv
    escapes it in the HTML and `convert_charrefs` resolves it again, so the body text holds
    literal tags. The corpus checks have to allow that: a tag-shaped string can only have
    come from the paper, since the extractor emits text nodes and formula alttext and never
    tags.
    """
    html = _page('<p class="ltx_p">An example: &lt;b&gt;bold&lt;/b&gt; and &lt;br&gt;.</p>')

    text = extract_text(html)

    assert "<b>bold</b>" in text


def test_raises_when_the_body_marker_is_missing() -> None:
    """A missing body marker must raise rather than degrade silently.

    Degrading means starting at the top of the page and writing licence and navigation
    text into the corpus, again without a signal. An arXiv redesign turns this red, which
    is exactly when someone should look.
    """
    html = (
        "<html><body><p>No title marker here.</p>"
        '<section id="bib"><ul><li>ref</li></ul></section></body></html>'
    )

    with pytest.raises(ValueError, match="ltx_title_document"):
        extract_text(html)


# --- math ---------------------------------------------------------------------


def test_keeps_math_alttext() -> None:
    """Keep the alttext and drop the MathML fragments: dropping the element breaks the
    sentence, keeping MathML adds a stream of `mi`/`mo` glyph pieces."""
    html = _page(
        '<p class="ltx_p">a neighborhood of size '
        '<math alttext="r"><semantics><mi>r</mi><annotation>x</annotation></semantics></math>'
        " in the input</p>"
    )

    text = extract_text(html)

    assert "of size r in the input" in text
    assert "semantics" not in text
    assert "annotation" not in text


def test_drops_math_that_sits_inside_a_dropped_element() -> None:
    """Math inside a dropped element must not be recovered; this pins the order of the
    two branches."""
    html = _page(
        '<figure><math alttext="SHOULD_NOT_APPEAR"><mi>x</mi></math></figure>'
        '<p class="ltx_p">Body.</p>'
    )

    text = extract_text(html)

    assert "SHOULD_NOT_APPEAR" not in text
    assert "Body." in text


def test_drops_nested_dropped_elements() -> None:
    """Dropping covers the whole subtree, not one level.

    A boolean in place of a stack would clear the drop when the inner element closes and
    then lose the rest of the document.
    """
    html = _page(
        '<figure><svg><path d="M0 0"></path></svg>figure caption</figure>'
        '<p class="ltx_p">Body survives.</p>'
    )

    text = extract_text(html)

    assert "figure caption" not in text
    assert "Body survives." in text


def test_recovers_from_malformed_tag_nesting() -> None:
    """Broken nesting must not take the rest of the body with it.

    Defensive: real arXiv pages are machine generated, so this guards against future
    edits rather than today's input. The unwinding in `handle_endtag` is what keeps a
    misplaced `</figure>` from leaving the drop stack permanently dirty; without it the
    output just gets shorter, with no other signal.
    """
    html = _page('<figure><svg></figure></svg><p class="ltx_p">Body survives.</p>')

    text = extract_text(html)

    assert "Body survives." in text


# --- regex boundaries ---------------------------------------------------------


@pytest.mark.parametrize(
    ("latex", "expected"),
    [
        # markup wrappers: drop the command, keep the content
        (r"\mathrm{softmax}", "softmax"),
        (r"\textsc{BASE}", "BASE"),
        (r"\mathbf{\mathrm{x}}", "x"),  # one level of nesting
        # symbol table
        (r"\theta", "θ"),
        (r"\cdot", "·"),
        (r"O(n/r)", "O(n/r)"),  # no commands, kept as is
        # name collisions: longest match must win
        (r"\inR^{H}", "R^{H}"),  # must not become "∈R^{H}"
        (r"x \in R", "x ∈ R"),
        # brackets and spacing
        (r"\left(x\right)", "(x)"),
        (r"a \, b", "a b"),
        # escaped characters
        (r"100\%", "100%"),
    ],
)
def test_latex_to_text(latex: str, expected: str) -> None:
    assert _latex_to_text(latex) == expected


def test_symbol_matching_survives_a_subscript() -> None:
    """`\\beta` inside `\\beta_{1}` must be replaced.

    Regression: the symbol regex ended with `\\b`, and an underscore counts as a word
    character, so there was no boundary after `\\beta` and the command was skipped.
    Subscripted symbols are the most common form in papers, so a lot went missing
    unnoticed.
    """
    assert _latex_to_text(r"\beta_{1}") == "β_{1}"
    assert _latex_to_text(r"\eta") == "η"


def test_symbol_matching_does_not_eat_the_start_of_a_longer_command() -> None:
    """The other direction: `\\thetaX` is an unknown command, not a symbol followed by X."""
    assert _latex_to_text(r"\thetaX") == r"\thetaX"


def test_nested_frac_is_left_alone() -> None:
    """Cleaning boundary: nested `\\frac` is kept as is.

    Not a limitation but a decision: half-cleaned text is worse than uncleaned text,
    because the corpus then has two spellings and no single rule can find either. This
    test pins the decision against a future one-level-only rule.
    """
    nested = r"\frac{1}{\sqrt{d_{k}}}"

    assert _latex_to_text(nested) == nested


# --- whitespace normalization -------------------------------------------------


def test_normalize_collapses_whitespace() -> None:
    assert _normalize_whitespace("a  \n\n\n  b\t\tc") == "a\n\nb c"


def test_normalize_removes_space_before_punctuation() -> None:
    """Removing a formula or citation marker leaves a space before the punctuation:
    "dilated convolutions [19], increasing" becomes "convolutions , increasing"."""
    assert _normalize_whitespace("convolutions , increasing") == "convolutions, increasing"


def test_normalize_keeps_space_before_an_opening_bracket() -> None:
    """Only trailing punctuation is cleaned; the space in "word (" is correct."""
    assert _normalize_whitespace("a function (see below)") == "a function (see below)"


def test_normalize_handles_non_breaking_space() -> None:
    """`\\xa0` is common in arXiv HTML; it looks exactly like a space but survives into
    the chunk text and skews token counts."""
    assert _normalize_whitespace("a\xa0b") == "a b"


# --- version ------------------------------------------------------------------


def test_extract_version() -> None:
    assert extract_version(_page("<p>x</p>")) == "arXiv:1234.56789v2 [cs.CL] 01 Jan 2024"


def test_extract_version_returns_empty_when_absent() -> None:
    """A missing version is not an error; the manifest is only a record and must not make
    the fetch fail."""
    assert extract_version("<html><body>no watermark</body></html>") == ""


# --- licence ------------------------------------------------------------------
#
# The licence never reaches the corpus files: it sits above the body marker, so the
# extractor drops it by design, which leaves sources.json as the only attribution record
# anywhere in the repository. arXiv renders the two kinds with different markup, and
# getting one wrong is invisible -- the field is simply empty, exactly like a paper whose
# licence could not be read.


def test_extract_license_reads_the_plain_link() -> None:
    html = (
        '<div class="abs-license">'
        '<a href="http://arxiv.org/licenses/nonexclusive-distrib/1.0/" '
        'title="Rights to this article">view license</a></div>'
    )

    assert extract_license(html) == "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"


def test_extract_license_reads_a_creative_commons_link_nested_in_a_span() -> None:
    """The CC form puts an `<img>` and a `<span>` between the `<a>` and the link text.

    Regression, and the first version failed silently here: a pattern expecting "view
    license" to be the link's immediate content matched the non-exclusive licence and
    returned "" for every CC paper. That is the worst way round, because the CC papers are
    the ones with the strictest terms and the ones a reader is most likely to be checking,
    and an empty field cannot be told from an unread one.
    """
    html = (
        '<div class="abs-license">'
        '<a href="http://creativecommons.org/licenses/by/4.0/" title="Rights to this article" '
        'class="has_license"> <img alt="license icon" role="presentation" '
        'src="https://arxiv.org/icons/licenses/by-4.0.png"/> <span>view license</span> </a></div>'
    )

    assert extract_license(html) == "http://creativecommons.org/licenses/by/4.0/"


def test_extract_license_ignores_the_footer_copyright_link() -> None:
    """The footer links to arXiv's licence *policy*, which is not this paper's licence.

    Matching any href containing "license" would record a help page as the paper's terms.
    """
    html = '<a href="https://info.arxiv.org/help/license/index.html">Copyright</a>'

    assert extract_license(html) == ""


def test_extract_license_returns_empty_when_absent() -> None:
    """A page with no licence link is not an error: the manifest is a record, and losing a
    paper over a metadata request would trade a usable corpus for a complete manifest."""
    assert extract_license("<html><body>no licence here</body></html>") == ""


# --- sanity check while fetching ----------------------------------------------
#
# `fetch_paper` needs the network, so these use a fake client. What is being tested is the
# check after the download: a body that is far too short must raise rather than produce a
# file that looks successful while holding almost nothing.


class _FakeClient:
    """Implements only the two methods `fetch_paper` uses."""

    def __init__(self, html: str) -> None:
        self._html = html

    def get(self, url: str) -> "_FakeResponse":
        return _FakeResponse(self._html)


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        pass


def test_fetch_paper_rejects_a_suspiciously_short_extraction() -> None:
    short_page = _page('<p class="ltx_p">Only a tiny bit of body text.</p>')

    with pytest.raises(ValueError, match="characters"):
        fetch_paper(_FakeClient(short_page), PAPERS[0])


def test_fetch_paper_accepts_a_full_length_extraction() -> None:
    """The other direction: a normal-length body must not trip the check.

    Testing only the rejection case would also pass with an infinite threshold, after
    which the script could never fetch anything at all.
    """
    long_page = _page(f'<p class="ltx_p">{"word " * 4000}</p>')

    text, version, url = fetch_paper(_FakeClient(long_page), PAPERS[0])

    assert len(text) > 10_000
    assert version == "arXiv:1234.56789v2 [cs.CL] 01 Jan 2024"
    assert url == "https://arxiv.org/html/1706.03762"


# --- real corpus --------------------------------------------------------------
#
# These read the committed files under eval/corpus/ and need no network; they check that
# the data checked in today is clean. HTML leftovers are invisible when reading a file and
# only surface later, as worse retrieval results.


def _corpus_files() -> list[Path]:
    return sorted(p for p in CORPUS_DIR.glob("*.txt"))


def test_corpus_directory_is_not_empty() -> None:
    """Without files the parametrized tests below would pass vacuously, so this closes
    that hole first."""
    assert _corpus_files(), f"{CORPUS_DIR} has no corpus files; run scripts/fetch_corpus.py first"


# Markers that would mean body text is not body text.
#
# `</` is deliberately absent, and its absence is the point. A tag-shaped string in the
# output is not by itself evidence of a leak: the extractor appends only from handle_data,
# which receives text nodes, and from a formula's alttext -- real tags are never emitted.
# So a "<" in the output arrived as a "&lt;" in the source, which means it was in the
# paper. T5's appendix quotes web documents from C4 that carry their own `<b>` and `<br>`
# markup; those 32 occurrences are the only ones across all twenty files, and the first
# version of this list failed on them while the extraction was in fact correct.
#
# What remains is markup only arXiv generates. It cannot arrive by that route, so finding
# it would mean the parser started emitting structure rather than text -- insurance against
# a rewrite of the extractor rather than against the version above.
#
# `&amp;` and `&#` are entity leftovers: convert_charrefs resolves those in text, so seeing
# one means it was double-escaped in the source.
#
# `<` and `>` alone are not searched for at all: both are ordinary math symbols here
# (`k<n`, `s_{i,j}>s_null`), and `k<n` is shaped exactly like a tag with a name.
_HTML_LEFTOVERS = ("<p>", "<p ", "<div", "<span", "<table", "<tr", "<td",
                   "ltx_", "href=", "xmlns", "class=", "&amp;", "&#")


@pytest.mark.parametrize("path", _corpus_files(), ids=lambda p: p.stem)
def test_corpus_has_no_html_leftovers(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    for marker in _HTML_LEFTOVERS:
        assert marker not in text, f"{path.name} still contains {marker!r}"


@pytest.mark.parametrize("path", _corpus_files(), ids=lambda p: p.stem)
def test_corpus_is_plausibly_sized(path: Path) -> None:
    """Each paper's body should be tens of thousands of characters.

    Much less means the body boundary failed and only a fragment was collected; much more
    means the references or something else came along.
    """
    text = path.read_text(encoding="utf-8")

    assert 10_000 < len(text) < 200_000, f"{path.name} is {len(text):,} characters, which is implausible"


@pytest.mark.parametrize("path", _corpus_files(), ids=lambda p: p.stem)
def test_corpus_has_no_mathml_fragments(path: Path) -> None:
    """MathML expands to tag names such as `mi`/`mo`/`mn`.

    Their presence means the formula alttext was not used and the MathML structure was
    collected instead, which is what the cleaning is meant to prevent.
    """
    text = path.read_text(encoding="utf-8")

    for fragment in ("<mi", "<mo", "<mn", "semantics"):
        assert fragment not in text, f"{path.name} contains MathML fragment {fragment!r}"


def test_sources_manifest_matches_the_files() -> None:
    """The manifest must match the files: it is the only record of which version of the
    corpus is checked in."""
    manifest = json.loads((CORPUS_DIR / "sources.json").read_text(encoding="utf-8"))

    recorded = {entry["file"] for entry in manifest["papers"]}
    assert recorded == {path.name for path in _corpus_files()}

    for entry in manifest["papers"]:
        assert entry["version"], f"{entry['slug']} has no recorded version"
        assert entry["characters"] > 10_000


def test_every_manifest_entry_records_a_licence() -> None:
    """Every arXiv paper has a licence, so an empty one means the read failed rather than
    that there is nothing to record.

    This is what would have caught the pattern that matched only the non-exclusive
    licences. The corpus files themselves carry no attribution -- the extractor removes it
    along with the rest of the page chrome -- so if the manifest is silent too, nothing in
    the repository says what these texts may be used for.
    """
    manifest = json.loads((CORPUS_DIR / "sources.json").read_text(encoding="utf-8"))

    missing = [entry["slug"] for entry in manifest["papers"] if not entry.get("license_url")]

    assert missing == []


def test_papers_declared_in_the_script_are_all_present() -> None:
    """Every paper declared in the script needs a corpus file; a missing one quietly
    shrinks the evaluation corpus."""
    on_disk = {path.stem for path in _corpus_files()}
    declared = {paper.slug for paper in PAPERS}

    assert declared <= on_disk, f"missing corpus files: {declared - on_disk}"
