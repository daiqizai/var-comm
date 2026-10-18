# Exact immutable-format continuation

The default staged whitespace check genuinely returned 2. Inspection found only
four unchanged model-constructor stdout lines and 2,224 legal trailing spaces
in four Matplotlib SVG path exports. This continuation verifies exact paths,
counts and bytes, retains the original failed checks, and applies one process-
local `-blank-at-eol` exception to those eight immutable files only. Every other
staged path uses the default check. No global Git, filter or file byte changes.

Original repository verification and 264 CPU checks run once after these checks,
then the normal exact staged commit. The initial interrupted copy and repaired
copy evidence remain preserved. No scientific or statistical entry is called.
