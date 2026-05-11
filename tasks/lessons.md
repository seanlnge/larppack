# Lessons

## Typography preferences

- When using the TikTok variable font in this project, explicitly set the variable weight axis to medium (`wght` ~500) instead of relying on default weight.
- When users ask for "same margin top and bottom" in text cards, account for font glyph bounding-box offsets (`bbox[1]`) so rendered vertical padding is visually symmetric.
- Keep typography settings aligned to the latest explicit user preference (e.g., semibold supersedes medium).
- Apply the same bbox-based vertical padding logic to rotated website stickers, not just main text cards.
- For Google Drive integration, support explicit OAuth variable names (`GOOGLE_DRIVE_OAUTH_CLIENT_ID` / `GOOGLE_DRIVE_OAUTH_CLIENT_SECRET`) and provide a built-in local callback URL.
- Do not expose raw `.env` editing in the UI when structured key-specific inputs are requested; use explicit fields with masked inputs and show/hide toggles.
- Avoid `multipart/form-data` for Drive `uploadType=multipart`; use a two-step flow (metadata create + media upload) to prevent Google parse errors.
- If users provide a Drive folder name where a folder ID is expected, auto-resolve/create the folder and persist the resolved ID to `.env`.
- When users request “upload individual images inside a named folder,” upload slide PNGs directly and create a Drive folder named after the output directory instead of zipping.
- Never call Flask context-bound helpers like `url_for()` inside background threads; compute URLs in the request handler and pass plain strings into the job runner.
- For LLM script generation quality issues (repetitive hooks like "7 ways..."), add explicit anti-pattern constraints + hook archetype rotation + recent-title anti-repeat context in the prompt.
- For repeated image selection across runs, combine controlled stochasticity (top-K sampling + temperature + score jitter) with penalties for images used in recent manifests of the same input stem.
- On Drive API `401 Invalid Credentials`, auto-refresh the access token and retry once before surfacing an error to the user.
