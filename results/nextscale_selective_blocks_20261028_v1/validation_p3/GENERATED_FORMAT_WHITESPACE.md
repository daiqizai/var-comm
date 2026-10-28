# Exact generated-artifact whitespace review

The first staged diff check stopped on trailing separators emitted by Matplotlib SVG path serialization and blank context lines in the preserved unified diff. All scientific source, values and figure geometry are unchanged. Four exact .gitattributes entries permit these formatting bytes only (two SVGs, the original patch, and the original failure transcript). No code-file or global whitespace rule was relaxed. Original failed check and subsequent successful review are retained.
