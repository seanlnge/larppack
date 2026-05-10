# Lessons

## Typography preferences

- When using the TikTok variable font in this project, explicitly set the variable weight axis to medium (`wght` ~500) instead of relying on default weight.
- When users ask for "same margin top and bottom" in text cards, account for font glyph bounding-box offsets (`bbox[1]`) so rendered vertical padding is visually symmetric.
- Keep typography settings aligned to the latest explicit user preference (e.g., semibold supersedes medium).
- Apply the same bbox-based vertical padding logic to rotated website stickers, not just main text cards.
- For Google Drive integration, support explicit OAuth variable names (`GOOGLE_DRIVE_OAUTH_CLIENT_ID` / `GOOGLE_DRIVE_OAUTH_CLIENT_SECRET`) and provide a built-in local callback URL.
