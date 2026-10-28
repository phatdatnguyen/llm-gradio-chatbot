# Studio tools

[Back to README](../Readme.md) | [RAG Chat guide](rag-guide.md)

The app provides **Summary**, **Spreadsheet**, **Presentation**, **Plot**, **Mind Map**, **Flashcards**, **Quizzes**, **Compare**, **Evidence Check**, saved tool
outputs, and the **Plugins** manager. Audio and video are outside this rollout.

Select the source checkboxes in RAG Chat, then open a tool under **Tools**. Tools
use the active working directory, collection, and local model selected in Prompt.
First choose LM Studio, Ollama, textgen, or KoboldCpp in **Connections** and press **Connect**. Both
chat tabs use that session's connected model choices. Model-native thinking stays
configured in the inference server. Textgen Studio generation requires the supported
`llama.cpp` loader. Both textgen and KoboldCpp need an explicitly selected
LM Studio/Ollama embedding connection for indexing the selected sources.
KoboldCpp Studio generation uses native schema-constrained grammar.

These nine tools use the connected server's schema-constrained generation, restrict citations to the
sources supplied in each request, and validate the output locally. A model can
support chat yet fail a complex structured task. Invalid JSON,
schema violations, refusals, and incomplete replies fail without a weaker-format
retry or replacement of the last successful result. Smaller local contexts can
require fewer selected sources, shorter outputs, or fewer requested slides/cards.
Errors identify the failing stage, such as source review, evidence consolidation,
or final output; consolidation errors also identify the round and batch. The previous saved output
stays available when a run fails.

- **Summary** reads all indexed passages in the selected documents in bounded
  batches, then assembles a cited summary. Give instructions for the desired focus,
  language, or format. It does not use the chat's small top-k retrieval window.
- **Spreadsheet** extracts requested fields into one row per selected document.
  Each generated cell retains its source references. Missing information is marked
  rather than filled with invented values. This stage extracts text values; it does
  not calculate formulas or generate executable analysis code.
- **Presentation** creates a cited slide deck with concise bullet points and speaker
  notes. Preview slides locally, then export an editable PowerPoint, Markdown, or JSON.
- **Plot** extracts reported numerical values and renders bar, line, or scatter
  charts locally, retaining the underlying data and source evidence.
- **Mind Map** organizes the selected documents into a central topic, branches, and
  related concepts. It reads every selected indexed passage through the same bounded
  evidence pipeline and retains supporting source IDs for its concepts.
- **Flashcards** turns selected documents into question-and-answer cards with source
  references. Reveal each answer and move between cards without another model request.
- **Quizzes** creates multiple-choice questions with four options, one correct answer,
  and a cited explanation. Checking answers and calculating scores run locally.

## Summary

Tools show progress and source coverage. Summary starts with the instruction
"Please help me summarize the documents" and shows the result in a scrollable
preview. Its narrow right column contains References and export controls. Click
**Edit** to replace the preview with a text editor, then **Save** to return to the
preview. Summary keeps one current result per workspace, saved after generation
and after editing, and restores it automatically when you reopen that workspace.
There is no Summary history selector or version-creation button. Existing older
Summary artifacts remain on disk; the latest is adopted if no current result exists.

## Spreadsheet

Spreadsheet starts with "Please help me to generate a spreadsheet for the documents".
Its editable table is on the left, with References above export controls in the
narrow right column. Both tools keep source excerpts in a bounded, scrollable box.
Filename, Export format, and Export each occupy a separate row.
The generated Spreadsheet is saved automatically and restored when you reopen the
workspace, including its original references. Edit the dataframe and click **Save**
below it to update that one output. Generating again replaces it. There is no history
dropdown or Load button. Older artifacts remain on disk; the latest is adopted only
when no current Spreadsheet exists.

## Presentation

Presentation appears before Plot and Mind Map in both the tool tabs and the
Plugins table. Its default instruction is "Please help me create a presentation
from the documents". The **Number of slides** slider selects **1–20 slides**, default
**10**, including any introduction or conclusion. Generation checks the exact count
and requires every selected document to be represented. Slide counts and readable
bullet lengths are constrained during generation. Use instructions to specify the
audience, language, and focus.

The left column contains a bounded slide preview, navigation controls, and speaker
notes. The right column contains References with scrollable excerpts, followed by
Filename (default **Presentation**), Export format, and Export on separate rows.
One current deck is saved automatically under `artifacts/presentation/current.json`
and restored when the workspace opens. Generate again to replace it; navigation
does not send another model request. **PPTX** is the default export, with Markdown
and JSON also available. Export does not change the saved deck.

PPTX exports use editable text on widescreen slides, with numbered citations on each
slide and source details in the speaker notes. They use the existing `python-pptx`
package, loaded only when exporting. Office is not required. Missing PPTX support
does not prevent generation, preview, or Markdown/JSON exports. This first version
creates text slides; it does not generate images, charts, animations, or native
PowerPoint equation objects. Revise the exported PPTX in your presentation editor,
or change the instructions and regenerate in the app.

## Plot

Plot appears immediately before Mind Map in both the tool tabs and the Plugins table.
It starts with "Please help me plot numerical data from the documents" and a chart
type selector: **Auto**, **Bar**, **Line**, or **Scatter**. Use the instructions to
specify the quantities or categories to compare. **Auto** chooses a supported
chart type from those instructions; choosing Bar, Line, or Scatter fixes the type.
A custom title and grouping column are optional. It uses the Prompt model and
selected sources without changing chat history.

Plots support **10,000 total points** across up to **six series**. For one checked
CSV or TSV file, the app asks the model to select the x, y, and optional grouping
columns, then reads every data row locally. Repeated coordinates are preserved;
the model does not rewrite or sample the dataset. For example, select IrisFlower.csv
and request: "Make a scatter plot with sepal length on the x-axis, sepal width on
the y-axis, colored by species."

This direct-table mode needs one header row with unique, nonempty column names,
numeric axis values, and at most 128 columns. It accepts files up to 8 MiB and checks
the original against the indexed copy. Filtering, aggregation, calculated columns,
unit conversion, and combining tables are not yet supported in this mode; prepare
those changes in a separate CSV before indexing it. Missing or invalid values fail
with a row/column message instead of silently removing rows. Other document formats
use the bounded evidence extraction pipeline and may reach its model output budget
before the 10,000-point limit.

Each plotted point retains a source citation and a short quotation. The app checks
that the quotation occurs in a cited source and that the numerical values occur in
that quotation. These checks do not prove that the model paired the right x/y values
or interpreted labels and units correctly; review the data table and source excerpts.
Numeric parsing accepts decimal and scientific notation with numbers separated from
units (for example, `10 nm`). Ambiguous notation is rejected rather than guessed.
The first version uses reported values without unit conversion, interpolation,
fitted curves, or digitization of chart images. Sources without compatible numerical
data are recorded as cited limitations. A selection with no usable values fails
without replacing the saved plot.

The left column contains the plot, a read-only preview of the first 200 data points,
and limitations. The chart and CSV/JSON exports retain the complete dataset.
The right column has References with a scrollable excerpt, then Filename (default
**Plot**), Export format, and Export on separate rows. One current result is saved
automatically at `artifacts/plot/current.json` and restored when the workspace opens.
Opening the Plot tab reloads this saved result and redraws its preview locally.
Plot retains its last successful result if a later generation fails;
opening a tab during generation leaves the running job alone.
Export **PNG** (default), **SVG**, **PDF**, **CSV**, or **JSON**. CSV/JSON retain the
data and supporting source/provenance information. Preview and export run locally
without additional model calls. Dense charts use fewer tick labels for readability;
this does not remove data points.

Plot uses [Matplotlib](https://matplotlib.org/stable/gallery/user_interfaces/web_application_server_sgskip.html)
through a fixed local renderer, with no generated code execution or external LaTeX.
Its dependency is isolated in `plugins/plot/requirements.txt`, included by the main
requirements file for normal installation. To add it to an existing environment:

```sh
python -m pip install -r plugins/plot/requirements.txt
```

The Plugins table reports whether the renderer is installed. Missing Matplotlib
does not prevent app startup, viewing the numeric data, or CSV/JSON exports;
chart previews and PNG/SVG/PDF exports require it.

## Mind Map

Mind Map starts with "Please help me create a mind map of the documents". Its map
is on the left in a scrollable viewport with a zoom control. The narrow right column
contains Filename (default **Mind Map**), Export format, and Export on separate rows.
There is no References panel. One current map is saved automatically per workspace
and restored when you reopen it; generating again replaces that map. The first version
supports generation and viewing; change the instructions and regenerate to revise it.
Maps have up to 60 nodes and five levels, with concise labels and descriptions.
Invalid trees or unknown source references are rejected before a map is displayed.
The app assigns internal node IDs automatically, so model-generated numeric,
non-English, or long identifiers do not prevent a valid map from being generated.

## Flashcards and Quizzes

Flashcards starts with "Please help me create flashcards from the documents";
Quizzes starts with "Please help me create a quiz from the documents". The slider
to the right of each instruction box sets the number to generate: **1–60 flashcards**
or **1–40 quiz questions**, both defaulting to **20**. The selected count takes
precedence over counts written in instructions. The generation schema requires
exactly this many items, and the app checks the count again before replacing the
saved output. It never pads or truncates a set to meet the count. Use instructions
to specify a topic, language, or difficulty. Both process all selected indexed
passages and require source citations for every card or question. Each selected
document must be represented, with items combining documents when needed.

Both learning tools use a study area on the left and References plus export controls
on the right. Flashcards hides each answer until **Show answer**. In Quizzes, each
option occupies a full row, labeled **A.**, **B.**, **C.**, or **D.** Select an option
and click **Check answer** to see feedback, an explanation, and your score.
Each question counts once per attempt; **Restart quiz** clears the attempt. Navigation
and quiz answers belong to the browser session and are not saved with the output.
Opening a workspace or generating a replacement starts a fresh study session.

One output per learning tool is saved automatically in each workspace and restored
when it reopens. Generate again to replace the set. Both tools export the complete
set to Markdown, CSV, or JSON, with answers, references, and provenance. CSV cells
are protected against spreadsheet formula execution. Quizzes also exports these
formats for import into a Moodle question bank:

- **Aiken (.txt)** is the default. It contains questions, four lettered options,
  and answer keys. Multiline text becomes one line; explanations and references
  are omitted to keep the file compatible with the [Aiken format](https://docs.moodle.org/en/Aiken_format).
- **Moodle XML (.xml)** includes questions, answer keys, explanations, and cited
  source evidence as general feedback. In Moodle's question-bank import screen,
  choose [Moodle XML format](https://docs.moodle.org/en/Moodle_XML_format); choose
  Aiken format for `.txt` exports.

Exports do not replace the saved result; JSON retains the complete source snapshots
and provenance. No additional packages are required. Learning questions may still
need review: citation validation checks where an answer came from, not whether the
model interpreted it correctly.

## Editing and exporting

Choose an output filename (default **Summary** or **Spreadsheet**) and export format,
then click **Export** to download directly. Export includes current spreadsheet cell
edits and does not save them; use **Save** to keep edits for next time. All tools
retain their original source snapshots independently of today's checkbox selection.
Edits are recorded as user edits; references from the model do not automatically
verify new text you add. Tool outputs remain separate from chatbot conversations.

Summary exports Markdown, JSON, and DOCX. Spreadsheet exports CSV, JSON, and XLSX.
Mind Map exports SVG, a Markdown outline, and JSON. SVG uses a local renderer with
escaped text, without executing model-generated markup or loading a remote script.
Markdown and JSON retain source/provenance information even though the map's UI
has no References panel. Exporting a map does not create another saved version.
DOCX and XLSX writers load only when requested and use the existing `python-docx`
and `openpyxl` dependencies. Exported spreadsheet text is protected against formula
execution. No additional packages or Office installation are needed for this stage.
DOCX exports use basic headings and paragraphs; equation markup remains text rather
than Word equation objects. XLSX rejects text beyond Excel's per-cell limit instead
of silently cutting it off; CSV and JSON remain available for longer values.

## Processing limits and cancellation

Generation sends selected indexed text to the connected local inference server
and can make several requests. Evidence batches use the loaded context remaining
after instructions, schemas, citation metadata, and output space are reserved.
There is no fixed 6,000-token input ceiling. Whole-document requests allow up to
32,768 output tokens, reduced for smaller contexts. These budgets are independent
of the provider-specific `*_MAX_OUTPUT_TOKENS` ordinary-chat cap and use conservative UTF-8 estimates without
tokenizer downloads. Whole-document generation uses text/OCR already indexed and does not run new
vision extraction. It is bounded to 40 documents, 512 chunks, and 500,000 source characters;
larger selections must be split. Oversized input is rejected before generation,
without silently omitting sources. Coverage means the indexed passages were processed,
not that every statement in the originals appears in the final summary. Citation
validation checks source membership, not factual entailment. Direct CSV/TSV Plot is
the exception: it reads a hash-verified original locally, with a separate budget of
10,000 indexed chunks, 8 MiB of source text, and a 16 MiB saved output. Only headers
and three sample rows are sent for column selection, together with your instructions.

Advanced settings can be set before launch; restart after changing them.

| Variable | Default and effect |
|---|---|
| `STUDIO_MAX_INPUT_TOKENS` | Unset or `0`: use available context. A positive value up to 1,048,576 imposes a lower evidence-input cap. |
| `STUDIO_MAX_CONSOLIDATION_ROUNDS` | `20`; accepts 1-100 rounds. |
| `STUDIO_MAX_MODEL_CALLS` | `256`; accepts 1-10,000 total model requests per run. |

Higher round/request limits allow longer runs when consolidation keeps making
progress. They do not enlarge the model's loaded context. A round that cannot
reduce the evidence while retaining its citations stops with an explanatory error.

If the initial notes omit an evidence section, its complete original passage and
citation are retained. If consolidation omits an identifier, the original note and
its citations are retained. Invalid or duplicated identifiers still fail validation.
Consolidation keeps its processing limits and stops with a clear budget error if
the complete evidence cannot fit, without dropping sources or retrying automatically.

**Stop** prevents subsequent model calls and discards the pending result after an
in-flight request returns. Changing the workspace or source selection also invalidates
pending work. Runs are not automatically resumed or restarted after a failure.
Deleting or reindexing a source during generation invalidates the source snapshot.
Exports also discard stale results if the workspace, output, or export options
change while a file is being prepared, preserving an existing download.

## Plugins

The Plugins tab manages trusted tools shipped with this repository. Toggle the
checkboxes in its **Enabled** column to apply changes immediately. All other columns
are read-only. Enable/disable settings are
local to this app in `.cache/plugins.json`, apply across workspaces,
and survive restarts. Dependency checks report installed versions without importing
heavy exporters. The manager does not install arbitrary third-party code or run
package installation from the browser. Missing export packages leave basic
Markdown/CSV/JSON outputs available.

## Compare and Evidence Check

**Compare** reads 2–8 checked documents and presents a side-by-side comparison of
their findings, methods, or other aspects specified in your instructions. Each of
up to 12 aspects includes an entry for every selected document, citations for found
information, and a cited synthesis. These bounds are enforced during generation.
Synthesis references come from its selected document findings. Missing information
is marked explicitly. It
does not imply that a document proves the opposite or that the documents are directly
comparable; review the synthesis and source excerpts.

**Evidence Check** accepts 1–20 claims, one per line (up to 1,000 characters each),
and reads all checked sources to review them. Results distinguish **Supported**,
**Contradicted**, **Mixed**, and **Not found**, with a rationale and quotations.
The model selects original excerpts, and the app copies their exact text into the
result while preserving your original claim text. It reviews all indexed passages
together when they fit; otherwise, it reviews every source batch and combines all
selected excerpts for the final assessment. It never turns summarized notes into
quotations or silently removes selected excerpts to fit. If the candidate evidence
exceeds available context, the run reports that limit. The verdict is a model
assessment of the selected documents,
not independent verification of the claim. **Not found** means insufficient evidence
in these sources; it does not mean the claim is false.

Both tools use the local Prompt model; thinking is configured in the inference server. They show a scrollable report
and read-only table on the left, with References and vertical export controls on the
right. Each saves one current result at `artifacts/compare/current.json` or
`artifacts/evidence_check/current.json`, restoring it when the workspace opens.
Failed or cancelled runs leave the previous result intact. Export Markdown (default),
CSV, or JSON with source and generation details. No additional packages are required.

Each stage includes offline tests and portability checks. Glossaries, timelines,
study guides, and research-gap reports can begin as Summary templates.
